import os
import numpy as np
from numpy import linalg as la
from itertools import zip_longest
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier


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
		self.labels_mean = [self.getMean(joints) for joints in self.storage[self.n0]]
		self.Y = []
		self.X = []

	def run(self, limit=20):
		c = 0
		first_train = True
		clf = None
		for n, joints_group in self.storage.items():
			if n == self.n0:
				continue
			c += 1
			if c < limit:
				m = self.matrixCalc(joints_group)
				steps = min(len(m), len(m[0]))
				label_ind = [i for i in range(len(self.labels_mean))]
				joints_ind = [i for i in range(len(joints_group))]
				new_labels_mean = self.labels_mean[:]

				for _ in range(steps):
				    x, y = self.getMatrixArgMin(m)

				    ji = joints_ind[x]
				    joints = joints_group[ji]
				    
				    li = label_ind[y]
				    new_labels_mean[li] = self.getMean(joints)

				    self.writeData(n, li, joints)

				    m = np.delete(m, x, 0)
				    m = np.delete(m, y, 1)
				    joints_ind = np.delete(joints_ind, x, 0)
				    label_ind = np.delete(label_ind, y, 0)

				if len(joints_ind) > 0:
					for ji in joints_ind:
						joints = joints_group[ji]
						new_labels_mean.append(self.getMean(joints))				
						self.writeData(n, len(new_labels_mean) - 1, joints)

				self.labels_mean = new_labels_mean
			else:
				if first_train:
					clf = RandomForestClassifier(max_depth=10, random_state=0)
					X_train = np.array([self.getFeatures(j) for j in self.X])
					clf.fit(X_train, self.Y)
					first_train = False
				for joints in joints_group:
					X_test = self.getFeatures(joints)
					predict_label = clf.predict(np.array([X_test]))[0]
					self.writeData(n, predict_label, joints, add=False)

	def getFeatures(self, joints):
		mean = self.getMean(joints)
		features = [la.norm(j - mean) for j in joints]
		norm_ = max(features)
		for i in range(len(joints)):
			for j in range(len(joints)):
				if i == j:
					continue
				features.append(la.norm(joints[i] - joints[j]))
		features = [elem / norm_ for elem in features]
		return np.array(features)
	
	def matrixCalc(self, joints_group):
		means = [self.getMean(joints) for joints in joints_group]
		matrix = []
		for mean in means:
			matrix.append([la.norm(label_mean - mean) for label_mean in self.labels_mean])
		return np.array(matrix)

	def getMatrixArgMin(self, m):
		index = m.argmin()
		num = len(m[0])
		return index // num, index % num

	def getMean(self, joints: np.array):
		joints_sum = np.zeros(shape=2)
		counter = 0
		for joint in joints:
			if joint[0] > 0:
				joints_sum += joint
				counter += 1
		return joints_sum / counter

	def writeData(self, n: int, label: int, joints: np.array, add=True):
		if add:
			self.Y.append(label)
			self.X.append(joints)

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
	path = "../data/Scenario03-01/detections.txt"
	model = predModel(path)
	model.run(400)


def main():
	data_path = "../data"
	for d in os.listdir(data_path):
		path = os.path.join(data_path, d)
		path = os.path.join(path, "detections.txt")
		model = predModel(path)
		model.run()


if __name__ == "__main__":
	mainSimpleCase()
	# main()
