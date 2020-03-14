
import numpy as np

# https://www.youtube.com/watch?v=HYLyhXRp298
# https://www.youtube.com/watch?v=QY9NTVh-Awo&list=PLK6OXn46jYQlUi1MYIxqcShx4_6mfPZ6j

class Net:
	class Channel:
		def __init__(self, size, equilibrium):
			self.equilibrium = equilibrium  # E_c the voltage when the current is 0
			self.maximum_conductance = 1  # g_bar_c the maximum conductance of the channel fixme, neuron sensitivity to chanel
			self.fraction_open = np.zeros(size)  # g_c the fraction of the channels current from 0 to 1 at time t
			conductance = np.zeros(size)  # placeholder value never used

			weights = np.zeros(size, size)  # todo

		def getConductance(self):  # g_c * g_bar_c
			self.conductance = self.fraction_open * self.maximum_conductance
			return self.conductance

		def getNetPotential(self, membrane_potential):  # v_m - E_c, note it's been flipped for calculating the voltage membrane potential
			return self.equilibrium - membrane_potential

		def getCurrent(self, membrane_potential):  # g_c * g_bar_c * (v_m - E_c)  fixme
			return self.getConductance() * self.getNetPotential(membrane_potential)

	def __init__(self, size):
		self.size = size

		self.inhibitory = Net.Channel(self.size, -70)  # chloride Cl- GABA
		self.excitatory = Net.Channel(self.size, +55)  # sodium Na+ glutamate
		self.leak = Net.Channel(self.size, -70)  # potassium K+

		self.leak.fraction_open = np.full(size, 0.1)

		self.threshold = -55
		self.gain = 10
		self.default_membrane_potential = -80

		self.I_net = self.getNetCurrent()
		self.membrane_potential = self.getEquilibriumMembranePotential()
		self.activations = np.zeros(size)


	def getEquilibriumMembranePotential(self):
		i = self.inhibitory.getConductance() * self.inhibitory.equilibrium
		e = self.excitatory.getConductance() * self.excitatory.equilibrium
		l = self.leak.getConductance() * self.leak.equilibrium
		denominator = self.inhibitory.getConductance() + self.excitatory.getConductance() + self.leak.getConductance()
		return (i + e + l) / denominator  # fixme unhandled case (denominator == 0)

	def getNetCurrent(self):
		i = self.inhibitory.getCurrent(self.membrane_potential)
		e = self.excitatory.getCurrent(self.membrane_potential)
		l = self.leak.getCurrent(self.membrane_potential)
		return i + e + l

	def step(self, dtvm):
		x = np.maximum(0, self.gain * (self.membrane_potential - self.threshold))  # ReLU, https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
		self.act = x / (x + 1)  # rate coding
		self.fired = self.act > 0
		self.membrane_potential = np.vectorize(lambda x, y: self.default_membrane_potential if x else y)(self.fired, self.membrane_potential)  # potential PyTorch incompatibility
		# self.membrane_potential = np.where(self.fired, self.membrane_potential, np.full(self.size, self.default_membrane_potential))

		self.I_net = self.getNetCurrent()
		self.membrane_potential = self.membrane_potential + dtvm * self.I_net

if __name__ == "__main__":
	dtvm = 0.05
	net = Net(100)
	for i in range(40):
		net.step(dtvm)


