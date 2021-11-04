from densecrf_np.pairwise import SpatialPairwise, BilateralPairwise
from densecrf_np.util import softmax

class DenseCRF(object):

	def __init__(self, image):

		self.alpha = 80
		self.beta = 13
		self.gamma = 3
		self.spatial_weight = 3
		self.bilateral_weight = 10

		self.sp = SpatialPairwise(image, gamma, gamma)
		self.bp = BilateralPairwise(image, alpha, alpha, beta, beta, beta)

	def infer(self, unary_logits, num_iterations=5):
		q = softmax(unary_logits)

		for _ in range(num_iterations):
			tmp1 = unary_logits

			output = self.sp.apply(q)
			tmp1 = tmp1 + self.spatial_weight * output  # Do NOT use the += operator here!

			output = self.bp.apply(q)
			tmp1 = tmp1 + self.bilateral_weight * output

			q = softmax(tmp1)

		return q
