#pragma once

enum class HashTableType {
	Atomic64,
	TicketBoard,
	AccelerationHash,
	CompactAccelerationHash,
	IndividualChaining
};

enum class HashMethod {
    Murmur,
	MurmurXor,
    SlabHashXor,
    SlabHashBoostCombine,
    SlabHashSingle
};