#!/usr/bin/env python3
# vim: ts=4 sts=0 sw=0 tw=100 noet

from abc import ABC, abstractmethod
import array
from enum import IntEnum
import itertools
import os
import random
import sys

# import galois
import numpy as np

# Binary transmission channel models: BSC, GilbertModel, BEC

P = 0.05 # 0.0005
Q = 0.80

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
	return bytes(pack_bits())

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
		return f"BSC(p={self.p}, p_prime={self.p_prime})"

	def channelise(self, bits):
		choices = np.random.choice([0, 1], size=len(bits), p=[self.p, self.p_prime])
		out_gen = (bits[i] if choice else 1 - bits[i] for i, choice in enumerate(choices))
		return array.array("B", out_gen)

class BEC(Channel):
	# BEC - Bianry Erasure Channel
	#	- Receive sent bit with probability 1-p
	#	- Receive signal, that the bit did not arrive with probability p
	#		Assumption: If a bit did not arrive, then get random bit with porbability 0.5.
	def __init__(self, p=P):
		Channel.__init__(self)
		self.p, self.p_prime = p, 1 - p

	def __str__(self):
		return f"BEC(p={self.p}, p_prime={self.p_prime})"

	def channelise(self, bits):
		choices = np.random.choice([0, 1], size=len(bits), p=[self.p, self.p_prime])
		out_gen = (bits[i] if choice else random.getrandbits(1) for i, choice in enumerate(choices))
		return array.array("B", out_gen)

def main(data):
	if data is None:
		data =  b"\x12\x23\xf9\x28"
	bits = to_bits(data)
	channels = [BSC(), BEC()]
	print(data.hex(" "))
	for channel in channels:
		result = to_bytes(channel.channelise(bits))
		print(channel, f"{np.array_equal(data, result)=:}")
		print(result.hex(" "))
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
