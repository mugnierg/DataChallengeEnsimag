import os
import numpy as np
from numpy import linalg as la
from itertools import zip_longest
from collections import OrderedDict


class dataProc(object):
	def __init__(self, file_name: str):
		self.file_name = file_name
		self.storage = OrderedDict()

	def run(self):
		with open(self.file_name) as f:
			for line in f:
				self.handle(line)
		return self.storage

	def handle(self, line: str):
		line = line.strip()
		data = line.split()
		data = [int(num) for num in data]
		n = data[0]
		joints = self.getJoints(data[1:])
		if n not in self.storage:
			self.storage[n] = []
		self.storage[n].append(joints)

	def getJoints(self, data: list):
		joints = []
		iter_data = iter(data)
		for x, y in zip_longest(iter_data, iter_data):
			joints.append([x, y])
		return np.array(joints)


class predModel(object):
	def __init__(self, file_name: str):
		path = os.path.dirname(file_name) + "/pred.txt"
		self.wfile = open(path, "w")
		self.storage = dataProc(file_name).run()
		
		self.n0 = min(self.storage.keys())
		self.labels = self.storage[self.n0][:]
		self.max_labels = len(self.labels)

	def run(self):
		counter = 0
		for n, joints_group in self.storage.items():
			if n == self.n0:
				continue
			m = self.matrixCalc(joints_group)
			m_size = max(len(m), len(m[0]))
			steps = min(len(m), len(m[0]))
			label_ind = [i for i in range(len(self.labels))]
			joints_ind = [i for i in range(len(joints_group))]
			new_labels = self.labels[:]

			for _ in range(steps):
			    x, y = self.getMatrixArgMin(m)

			    ji = joints_ind[x]
			    joints = joints_group[ji]
			    
			    li = label_ind[y]
			    new_labels[li] = joints

			    self.writeData(n, li, joints)

			    m = np.delete(m, x, 0)
			    m = np.delete(m, y, 1)
			    joints_ind = np.delete(joints_ind, x, 0)
			    label_ind = np.delete(label_ind, y, 0)

			if len(joints_ind) > 0 and m_size > self.max_labels:
				counter += 1
				if counter > 5:
					counter = 0
					for ji in joints_ind:
						joints = joints_group[ji]
						new_labels.append(joints)				
						self.writeData(n, len(new_labels) - 1, joints)
					self.max_labels = len(new_labels)

			self.labels = new_labels
	
	def matrixCalc(self, joints_group):
		matrix = []
		for j1 in joints_group:
			val = [self.getSqrScore(j1, j2) for j2 in self.labels]
			matrix.append(val)
		return np.array(matrix)

	def getMatrixArgMin(self, m):
		index = m.argmin()
		num = len(m[0])
		return index // num, index % num

	def getSqrScore(self, joints1: np.array, joints2: np.array):
		num = len(joints1)
		score = 0
		for i in range(num):
			v = joints1[i] - joints2[i]
			score += np.dot(v, v)
		return score

	def writeData(self, n: int, label: int, joints: np.array):
		s = str(n) + ' ' + str(label+1) + ' 0 '
		data = []
		for joint in joints:
			data.append(str(joint[0]))
			data.append(str(joint[1]))
		s += ' '.join(data) + '\n'
		self.wfile.write(s)

	def __del__(self):
		self.wfile.close()


def mainSimpleCase():
	path = "../data/Scenario02-03/detections.txt"
	model = predModel(path)
	model.run()


def main():
	data_path = "../data"
	for d in os.listdir(data_path):
		path = os.path.join(data_path, d)
		path = os.path.join(path, "detections.txt")
		model = predModel(path)
		model.run()


if __name__ == "__main__":
	# mainSimpleCase()
	main()
