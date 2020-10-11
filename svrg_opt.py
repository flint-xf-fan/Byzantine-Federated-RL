  
from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import PerlmutterHvp
from rllab.optimizers.minibatch_dataset import BatchDataset
from rllab.misc.ext import sliced_fun
import tensorflow as tf
import time
from functools import partial
import pyprind
from numpy import linalg as LA
import numpy as np
from scipy.stats import norm
from copy import deepcopy

class SVRGOptimizer(Serializable):
    """
    Performs stochastic variance reduction gradient (SVRG) in TRPO
    """

    def __init__(
            self,
            eta=0.01,
            alpha=0.001,
            max_epochs=1,
            tolerance=1e-5,
            batch_size=32,
            epsilon=1e-8,
            verbose=False,
            num_slices=1,
            use_SGD=False,
            scale=1.0,
            backtrack_ratio=0.5,
            max_backtracks=10,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=1.,
            hvp_approach=None,
            max_batch=10,
            **kwargs):
        """
        :param max_epochs:
        :param tolerance:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will be used.
        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
        :param kwargs:
        :return:
        """
        Serializable.quick_init(self, locals())
        self._eta = eta
        self._alpha = alpha
        self._opt_fun = None
        self._target = None
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._epsilon = epsilon
        self._verbose = verbose
        self._input_vars = None
        self._num_slices = num_slices
        self._scale = scale
        self._use_SGD = use_SGD
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks
        self._max_batch = max_batch

        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        if hvp_approach is None:
            hvp_approach = PerlmutterHvp(num_slices)
        self._hvp_approach = hvp_approach

        logger.log('max_batch %d' % (self._max_batch))
        logger.log('mini_batch %d' % (self._batch_size))
        logger.log('cg_iters %d' % (self._cg_iters))
        logger.log('subsample_factor %f' % (self._subsample_factor))

    def update_opt(self, loss, loss_tilde, target, target_tilde,
                   leq_constraint, inputs, extra_inputs=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param loss_tilde: Symbolic expression for the loss function of w_tilde.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :policy network with parameter w to optimize
        :param target_tilde: A parameterized policy network with control variate w_tilde parameter.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """
        if extra_inputs is None:
            extra_inputs = list()
        self._input_vars = inputs + extra_inputs

        self._target = target
        self._target_tilde = target_tilde

        constraint_term, constraint_value = leq_constraint
        self._max_constraint_val = constraint_value

        w = target.get_params(trainable=True)
        grads = tf.gradients(loss, xs=w)
        for idx, (g, param) in enumerate(zip(grads, w)):
            if g is None:
                grads[idx] = tf.zeros_like(param)
        flat_grad = tensor_utils.flatten_tensor_variables(grads)

        w_tilde = target_tilde.get_params(trainable=True)
        grads_tilde = tf.gradients(loss_tilde, xs=w_tilde)
        for idx, (g_t, param_t) in enumerate(zip(grads_tilde, w_tilde)):
            if g_t is None:
                grads_tilde[idx] = tf.zeros_like(param_t)
        flat_grad_tilde = tensor_utils.flatten_tensor_variables(grads_tilde)

        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss,
            ),
            f_loss_tilde=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss_tilde,
            ),
            f_grad=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
            ),
            f_grad_tilde=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad_tilde,
            ),
            f_loss_constraint=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=[loss, constraint_term],
            ),
        )

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)
        self._hvp_approach.update_opt(f=constraint_term, target=target, inputs=inputs + extra_inputs,
                                      reg_coeff=self._reg_coeff)

    def loss(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)
        return self._opt_fun["f_loss"](*(inputs + extra_inputs))

    def loss_tilde(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)
        return self._opt_fun["f_loss_tilde"](*(inputs + extra_inputs))

    def optimize(self, inputs, extra_inputs=None,
                 subsample_grouped_inputs=None):

        if len(inputs) == 0:
            raise NotImplementedError

        f_loss = self._opt_fun["f_loss"]
        f_grad = self._opt_fun["f_grad"]
        f_grad_tilde = self._opt_fun["f_grad_tilde"]

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        param = np.copy(self._target.get_param_values(trainable=True))
        logger.log("Start SVRG CG subsample optimization: #parameters: %d, #inputs: %d, #subsample_inputs: %d" % (
            len(param), len(inputs[0]), self._subsample_factor * len(inputs[0])))

        subsamples = BatchDataset(
            inputs,
            int(self._subsample_factor * len(inputs[0])),
            extra_inputs=extra_inputs)

        dataset = BatchDataset(
            inputs,
            self._batch_size,
            extra_inputs=extra_inputs)

        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))
                progbar = pyprind.ProgBar(len(inputs[0]))
            # g_u = 1/n \sum_{b} \partial{loss(w_tidle, b)} {w_tidle}
            grad_sum = np.zeros_like(param)
            g_mean_tilde = sliced_fun(f_grad_tilde, self._num_slices)(
                    inputs, extra_inputs)
            logger.record_tabular('g_mean_tilde', LA.norm(g_mean_tilde))
            print("-------------mini-batch-------------------")
            num_batch = 0
            while num_batch < self._max_batch:
                batch = dataset.random_batch()
                # todo, pick mini-batch with weighted prob.
                if self._use_SGD:
                    g = f_grad(*(batch))
                else:
                    g = f_grad(*(batch)) - \
                            f_grad_tilde(*(batch)) + g_mean_tilde
                grad_sum += g
                subsample_inputs = subsamples.random_batch()
                Hx = self._hvp_approach.build_eval(subsample_inputs)
                self.conjugate_grad(g, Hx, inputs, extra_inputs)
                num_batch += 1
            print("max batch achieved {:}".format(num_batch))
            grad_sum /= 1.0 * num_batch
            if self._verbose:
                progbar.update(batch[0].shape[0])
            logger.record_tabular('gdist', LA.norm(
                grad_sum - g_mean_tilde))

            cur_w = np.copy(self._target.get_param_values(trainable=True))
            w_tilde = self._target_tilde.get_param_values(trainable=True)
            self._target_tilde.set_param_values(cur_w, trainable=True)
            logger.record_tabular('wnorm', LA.norm(cur_w))
            logger.record_tabular('w_dist', LA.norm(
                cur_w - w_tilde) / LA.norm(cur_w))

            if self._verbose:
                if progbar.active:
                    progbar.stop()

            if abs(LA.norm(cur_w - w_tilde) /
                   LA.norm(cur_w)) < self._tolerance:
                break

    def conjugate_grad(self, deltaW, Hx, inputs, extra_inputs=()):
        # s = H^-1 g
        descent_direction = krylov.cg(Hx, deltaW, cg_iters=self._cg_iters)
        init_step = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8)))
        # s' H s = g' s, as s = H^-1 g
        # init_step = np.sqrt(2.0 * self._max_constraint_val *
        #(1. / (descent_direction.dot(deltaW)) + 1e-8))
        if np.isnan(init_step):
            init_step = 1.
        descent_step = init_step * descent_direction
        return self.line_search(descent_step, inputs, extra_inputs)

    def line_search(self, descent_step, inputs, extra_inputs=()):
        f_loss = self._opt_fun["f_loss"]
        f_loss_constraint = self._opt_fun["f_loss_constraint"]
        prev_w = np.copy(self._target.get_param_values(trainable=True))
        loss_before = f_loss(*(inputs + extra_inputs))
        n_iter = 0
        succ_line_search = False
        for n_iter, ratio in enumerate(
                self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * descent_step
            cur_w = prev_w - cur_step
            self._target.set_param_values(cur_w, trainable=True)
            loss, constraint_val = sliced_fun(f_loss_constraint,
                                              self._num_slices)(inputs, extra_inputs)
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                succ_line_search = True
                break

        if (np.isnan(loss) or np.isnan(constraint_val) or loss >=
                loss_before or constraint_val >= self._max_constraint_val):
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(constraint_val):
                logger.log("Violated because constraint is NaN")
            if loss >= loss_before:
                logger.log("Violated because loss not improving")
            if constraint_val >= self._max_constraint_val:
                logger.log(
                    "Violated because constraint {:} is violated".format(constraint_val))

            self._target.set_param_values(prev_w, trainable=True)

        logger.log("backtrack iters: %d" % n_iter)
        return succ_line_search