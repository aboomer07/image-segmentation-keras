from .densecrf_np.pairwise import SpatialPairwise, BilateralPairwise
from .densecrf_np.util import softmax
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf

class CRFLayer(Layer):
	tf.compat.v1.enable_eager_execution()
	def __init__(self, num_iterations=5):
		super(CRFLayer, self).__init__()

		self.alpha = 80
		self.beta = 13
		self.gamma = 3
		self.spatial_ker_weight = 3
		self.bilateral_ker_weight = 10


		self.iterations = num_iterations

	def call(self, inputs):
		print(tf.executing_eagerly())
		unaries = inputs[0][0, :, :]
		unaries = unaries.numpy()
		rgb = inputs[1][0, :, :]
		rgb = rgb.numpy()

		self.rgb = np.resize(rgb, unaries.shape)

		self.sp = SpatialPairwise(self.rgb, self.gamma, self.gamma)
		self.bp = BilateralPairwise(self.rgb, self.alpha, self.alpha, 
			self.beta, self.beta, self.beta)

		q = softmax(unaries)

		for _ in range(self.iterations):
			tmp1 = inputs

			output = self.sp.apply(q)
			tmp1 = tmp1 + self.spatial_weight * output

			output = self.bp.apply(q)
			tmp1 = tmp1 + self.bilateral_weight * output

			q = softmax(tmp1)

		q = tf.convert_to_tensor(q)
		return(q)
