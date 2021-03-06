#!/usr/bin/env python3
# vim: ts=4 sts=0 sw=0 tw=100 noet

from abc import ABC, abstractmethod
import array
from enum import IntEnum
import itertools
import os
import random
import statistics as st
import sys

import galois
import numpy as np

# Binary transmission channel models: BSC, GilbertModel, BEC

P = 0.005
Q = 0.3

def to_bits(buf: memoryview, bits=None):
	if bits is None:
		bits = len(buf)*8
	class Byte:
		def __init__(self, x):
			self.x = x
		def __and__(self, other):
			return bool(self.x & other)
	m = bits >> 3
	r = bits & 0x7
	def extract_bits():
		Bits = lambda x: (x & 0x80, x & 0x40, x & 0x20, x & 0x10, x & 0x8, x & 0x4, x & 0x2, x & 0x1)
		for i, byte in enumerate(map(Byte, buf[: m+bool(r)])):
			byte = Byte(byte)
			yield Bits(byte) if i < m else Bits(byte)[:r]
	return array.array("B", itertools.chain.from_iterable(extract_bits()))

def to_bytes(bits):
	def pack_bits():
		for i in range(0, len(bits), 8):
			slice = bits[i:i + 8]
			res = 0
			for i in range(0, len(slice)):
				res <<= 1
				res |= slice[i]
			yield res
	return np.fromiter(pack_bits(), dtype=np.uint8)

class Channel(ABC):
	@abstractmethod
	def channelise(self, bits):
		...

class BSC(Channel):
	# BSC - Binary Symmetric Channel
	#	- Receive sent bit with probability 1-p
	#	- Receive opp. bit with probability p
	def __init__(self, p=P):
		Channel.__init__(self)
		self.p, self.p_prime = p, 1 - p

	def __str__(self):
		return f"BSC(p={self.p}, p'={self.p_prime})"

	def channelise(self, bits):
		choices = np.random.choice([0, 1], size=len(bits), p=[self.p, self.p_prime])
		out_gen = (bits[i] if choice else 1 - bits[i] for i, choice in enumerate(choices))
		return array.array("B", out_gen)

class GilbertModel(Channel):
	class State(IntEnum):
		CORRECT = 0
		INCORRECT = 1

	def __init__(self, ci, ic, pc=P, pi=Q):
		Channel.__init__(self)
		self.ci, self.ci_prime = ci, 1 - ci # ci: correct -> incorrect, ci_prime: correct -> correct
		self.ic, self.ic_prime = ic, 1 - ic # ic: incorrect -> correct, ic_prime: incorrect -> incorrect
		self.pc, self.pc_prime = pc, 1 - pc # pc: P(error) in correct state
		self.pi, self.pi_prime = pi, 1 - pi # pi: P(error) in incorrect state, pi >> pc
		self.perr = ((self.pc, self.pc_prime), (self.pi, self.pi_prime))
		self.state = self.State.CORRECT

	def __str__(self):
		return f"GilbertModel(ci={self.ci}, ci'e={self.ci_prime}, ic={self.ic}, ic'={self.ic_prime}, " \
			f"pc={self.pc}, pc'={self.pc_prime}, pi={self.pi}, pi'={self.pi_prime})"

	def step_state(self):
		if self.state == self.State.CORRECT:
			self.state = np.random.choice([self.State.INCORRECT, self.State.CORRECT], p=[self.ci, self.ci_prime])
		else:
			self.state = np.random.choice([self.State.CORRECT, self.State.INCORRECT], p=[self.ic, self.ic_prime])
		return self.state

	def channelise(self, bits):
		def choices():
			for i in range(len(bits)):
				yield np.random.choice([random.getrandbits(1), bits[i]], p=self.perr[self.step_state()])
		return array.array("B", choices())

class BEC(Channel):
	# BEC - Bianry Erasure Channel
	#	- Receive sent bit with probability 1-p
	#	- Receive signal, that the bit did not arrive with probability p
	#		Assumption: If a bit did not arrive, then get random bit with porbability 0.5.
	def __init__(self, p=P):
		Channel.__init__(self)
		self.p, self.p_prime = p, 1 - p

	def __str__(self):
		return f"BEC(p={self.p}, p'={self.p_prime})"

	def channelise(self, bits):
		choices = np.random.choice([0, 1], size=len(bits), p=[self.p, self.p_prime])
		out_gen = (bits[i] if choice else random.getrandbits(1) for i, choice in enumerate(choices))
		return array.array("B", out_gen)

def encoded_chunks(rs, data, chunk_size):
	return [
		to_bits(rs.encode(data[ib : ib+chunk_size]))
		for ib in range(0, len(data), chunk_size)]

def channels_stats(channels, chunks, rs, data_size, p):
	n, k = rs.n, rs.k
	r = n - k
	code_channel = f"RS({n}, {k}) p={p})"
	print(f"=== BEGIN: {code_channel} data size: {data_size} bytes, chunk size: {k}, number of chunks: {len(chunks)} ===")
	for i, channel in enumerate(channels):
		if i > 0:
			print()
		invalid = 0
		invalid_bytes = 0
		corrected = []
		for chunk in chunks:
			channelised = to_bytes(channel.channelise(chunk))
			decoded, errors = rs.decode(channelised, errors=True)
			if errors == -1:
				invalid += 1
				invalid_bytes += len(channelised) - r
			else:
				corrected.append(errors)
		print(channel)
		if len(corrected) == 0:
			corrected.append(0)
		print(f"corrected errors/chunk avg: {st.mean(corrected):.2f} std: {st.pstdev(corrected):.2f}")
		print(f"received: invalid chunks: {invalid}/{len(chunks)}, invalid bytes: {invalid_bytes}/{data_size}")
	print(f"=== END: {code_channel} ===\n")

def stats_change_control_positions(data):
	N, K = 255, (203, 213, 223, 233)
	channels = [BSC(), BEC(), GilbertModel(0.005, 0.2)]
	for k in K:
		rs = galois.ReedSolomon(N, k)
		chunks = encoded_chunks(rs, data, k)
		channels_stats(channels, chunks, rs, len(data), P)

def stats_change_channel_config(data):
	N, K = 255, 223
	P = (0.001, 0.003, 0.005, 0.01)
	rs = galois.ReedSolomon(N, K)
	chunks = encoded_chunks(rs, data, K)
	for p in P:
		channels = [BSC(p=p), BEC(p=p), GilbertModel(0.005, 0.2, pc=p)]
		channels_stats(channels, chunks, rs, len(data), p)

def main(data):
	if data is None:
		data =  b"\x12\x23\xf9\x28"
	ndarr = np.ndarray(len(data), dtype=np.uint8, buffer=data)
	print("\n\n========== Change number of control position ==========\n")
	stats_change_control_positions(ndarr)
	print("\n\n========== Change channel configuration ==========\n")
	stats_change_channel_config(ndarr)
	return

if __name__ == "__main__":
	data = None
	if os.getenv("FILE"):
		import hashlib
		import requests
		DATA_DIR = "data"
		if not os.path.isdir(DATA_DIR):
			os.mkdir(DATA_DIR)
		FILE_NAME, FILE_URL = "64mb", "https://ftp.atman.pl/64mb"
		file_path = os.path.join(DATA_DIR, FILE_NAME)
		buf = bytearray(64 << 20)
		if not os.path.isfile(file_path):
			with requests.get(FILE_URL, stream=True) as req:
				read = req.raw.readinto(buf)
				if read != len(buf):
					raise Exception(f"Read {read}/{len(buf)} bytes")
			with open(file_path, "wb") as fw:
				fw.write(buf)
		else:
			with open(file_path, "rb") as fr:
				fr.readinto(buf)
		print(f"{FILE_NAME}={hashlib.sha256(buf).hexdigest()}")
		data = memoryview(buf)[:int(os.getenv("FILE"))]
	sys.exit(main(data))
