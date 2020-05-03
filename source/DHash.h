#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <intrin.h>
#include <immintrin.h>
#ifndef BUILD_MSVC
#include <avxintrin.h>
#include <avx2intrin.h>
#include <bmi2intrin.h>
#endif

#define ASSERT(expr) if (!(expr)) abort();
#define RESTRICT __restrict

uint32_t _horizontal_sum_4xi32(__m128i vec)
{
	__m128i hi64 = _mm_unpackhi_epi64(vec, vec);
	__m128i sum64 = _mm_add_epi32(hi64, vec);
	__m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
	__m128i sum32 = _mm_add_epi32(hi32, sum64);
	return _mm_cvtsi128_si32(sum32);
}

uint32_t _horizontal_sum_8xi32(__m256i vec)
{
	__m128i lo_128 = _mm256_castsi256_si128(vec);
	__m128i hi_128 = _mm256_extracti128_si256(vec, 1);
	return _horizontal_sum_4xi32(_mm_add_epi32(lo_128, hi_128));
}

void Resize_3(const uint8_t* RESTRICT in, const int in_w, const int in_h, uint8_t* RESTRICT out)
{
	const int stride_w = in_w / 8;
	const int stride_h = in_h / 8;
	const int stride_w_a16 = stride_w / 16 * 16;
	const int num_px_per_block = stride_w * stride_h;
	const float inv_num_px_per_block = 1.0f / num_px_per_block;

	__m256 normalizing_multiplier = _mm256_broadcast_ss(&inv_num_px_per_block);
	const __m128i midpoint_u16 = _mm_set1_epi16(127);
	const __m128i midpoint_u8 = _mm_set1_epi8(127);

	for (int out_row = 0; out_row < 8; ++out_row)
	{
		__m128i block_accumulate_wide[8];
		for (int i = 0; i < 8; ++i)
			block_accumulate_wide[i] = _mm_setzero_si128();

		for (int in_row = out_row * stride_h, _end_row = in_row + stride_h; in_row < _end_row; ++in_row)
		{
			const int offset_in = in_row * in_w;

			__m128i row[8];
			for (int i = 0; i < 8; ++i)
				row[i] = _mm_setzero_si128();

			for (int block_i = 0; block_i < 8; ++block_i)
			{
				__m256i accumulate_1 = _mm256_setzero_si256();
				__m256i accumulate_2 = _mm256_setzero_si256();

				for (int in_col = block_i * stride_w + offset_in, _end_col = in_col + stride_w_a16; in_col < _end_col; in_col += 16)
				{
					__m128i lo = _mm_loadu_si128((__m128i*) & in[in_col]);
					__m256i lo_8x32 = _mm256_cvtepu8_epi32(lo);
					__m128i hi = _mm_loadu_si128((__m128i*) & in[in_col + 8]);
					__m256i hi_8x32 = _mm256_cvtepu8_epi32(hi);
					accumulate_1 = _mm256_add_epi32(accumulate_1, lo_8x32);
					accumulate_2 = _mm256_add_epi32(accumulate_2, hi_8x32);
				}
				__m256i sum = _mm256_add_epi32(accumulate_1, accumulate_2);
				row[block_i] = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
			}

			for (int i = 0; i < 8; ++i)
				block_accumulate_wide[i] = _mm_add_epi32(block_accumulate_wide[i], row[i]);
		}

		__m256i block_accumulate = _mm256_setzero_si256();
		for (int i = 0; i < 8; ++i)
			block_accumulate.m256i_i32[i] = _horizontal_sum_4xi32(block_accumulate_wide[i]);

		__m256 f_block_accumulate = _mm256_cvtepi32_ps(block_accumulate);
		__m256 f_block_accumulate_normalized = _mm256_mul_ps(f_block_accumulate, normalizing_multiplier);
		block_accumulate = _mm256_cvtps_epi32(f_block_accumulate_normalized);

		// Convert 8x32 to 8x16
		__m128i packed = _mm_packs_epi32(_mm256_castsi256_si128(block_accumulate), _mm256_extracti128_si256(block_accumulate, 1));
		packed = _mm_packus_epi16(packed, _mm_setzero_si128());

		// Store lower 8x8 to the output, we can write this in one instruction without any specific intrinsics.
		*(__int64*)&out[out_row * 8] = _mm_cvtsi128_si64(packed);
	}
}

void Resize_2(const uint8_t* RESTRICT in, const int in_w, const int in_h, uint8_t* RESTRICT out)
{
	const int stride_w = in_w / 8;
	const int stride_h = in_h / 8;
	const int stride_w_a16 = stride_w / 16 * 16;
	const int num_px_per_block = stride_w * stride_h;
	const float inv_num_px_per_block = 1.0f / num_px_per_block;

	__m256 normalizing_multiplier = _mm256_broadcast_ss(&inv_num_px_per_block);

	for (int out_row = 0; out_row < 8; ++out_row)
	{
		__m256i block_accumulate = _mm256_setzero_si256();

		for (int in_row = out_row * stride_h, _end_row = in_row + stride_h; in_row < _end_row; ++in_row)
		{
			const int offset_in = in_row * in_w;

			__m256i row = _mm256_setzero_si256();

			for (int block_i = 0; block_i < 8; ++block_i)
			{
				__m256i accumulate = _mm256_setzero_si256();
				for (int in_col = block_i * stride_w + offset_in, _end_col = in_col + stride_w_a16; in_col < _end_col; in_col += 16)
				{
					__m128i lo = _mm_loadu_si128((__m128i*) & in[in_col]);
					__m256i lo_8x32 = _mm256_cvtepu8_epi32(lo);
					__m128i hi = _mm_loadu_si128((__m128i*) & in[in_col + 8]);
					__m256i hi_8x32 = _mm256_cvtepu8_epi32(hi);
					__m256i sum = _mm256_add_epi32(hi_8x32, lo_8x32);
					accumulate = _mm256_add_epi32(accumulate, sum);
				}
				row.m256i_i32[block_i] = _horizontal_sum_8xi32(accumulate);
			}

			block_accumulate = _mm256_add_epi32(block_accumulate, row);
		}

		__m256 f_block_accumulate = _mm256_cvtepi32_ps(block_accumulate);
		__m256 f_block_accumulate_normalized = _mm256_mul_ps(f_block_accumulate, normalizing_multiplier);
		block_accumulate = _mm256_cvtps_epi32(f_block_accumulate_normalized);

		// Convert 8x32 to 8x16
		__m128i packed = _mm_packs_epi32(_mm256_castsi256_si128(block_accumulate), _mm256_extracti128_si256(block_accumulate, 1));
		packed = _mm_packus_epi16(packed, _mm_setzero_si128());

		// Store lower 8x8 to the output, we can write this in one instruction without any specific intrinsics.
		*(__int64*)&out[out_row * 8] = _mm_cvtsi128_si64(packed);
	}
}

void Resize_1(const uint8_t* RESTRICT in, const int in_w, const int in_h, uint8_t* RESTRICT out)
{
	const int stride_w = in_w / 8;
	const int stride_h = in_h / 8;
	const int stride_w_a16 = stride_w / 16 * 16;
	const int num_px_per_block = stride_w * stride_h;
	const float inv_num_px_per_block = 1.0f / num_px_per_block;

	for (int out_row = 0; out_row < 8; ++out_row)
	{
		const int offset_out = out_row * 8;

		for (int out_col = 0; out_col < 8; ++out_col)
		{
			__m256i _accumulate = _mm256_setzero_si256();

			for (int in_row = out_row * stride_h, _end_row = in_row + stride_h; in_row < _end_row; ++in_row)
			{
				const int offset_in = in_row * in_w;

				for (int in_col = out_col * stride_w + offset_in, _end_col = in_col + stride_w_a16; in_col < _end_col; in_col += 16)
				{
					__m128i lo = _mm_loadu_si128((__m128i*)&in[in_col]);
					__m256i lo_8x32 = _mm256_cvtepu8_epi32(lo);
					__m128i hi = _mm_loadu_si128((__m128i*)&in[in_col + 8]);
					__m256i hi_8x32 = _mm256_cvtepu8_epi32(hi);
					__m256i sum = _mm256_add_epi32(hi_8x32, lo_8x32);
					_accumulate = _mm256_add_epi32(_accumulate, sum);
				}
			}

			uint32_t accumulate = _horizontal_sum_8xi32(_accumulate);
			accumulate = (uint32_t)round((float)accumulate * inv_num_px_per_block);

			out[offset_out + out_col] = (uint8_t)accumulate;
		}
	}
}

void Resize_Naive(const uint8_t* RESTRICT in, const int in_w, const int in_h, uint8_t* RESTRICT out)
{
	const int stride_w = in_w / 8;
	const int stride_h = in_h / 8;
	const int stride_w_a16 = stride_w / 16 * 16;
	const int num_px_per_block = stride_w * stride_h;
	const float inv_num_px_per_block = 1.0f / num_px_per_block;

	for (int out_row = 0; out_row < 8; ++out_row)
	{
		const int offset_out = out_row * 8;

		for (int out_col = 0; out_col < 8; ++out_col)
		{
			uint32_t accumulate = 0;

			for (int in_row = out_row * stride_h, _end_row = in_row + stride_h; in_row < _end_row; ++in_row)
			{
				const int offset_in = in_row * in_w;
				for (int in_col = out_col * stride_w, _end_col = in_col + stride_w_a16; in_col < _end_col; ++in_col)
				{
					accumulate += in[offset_in + in_col];
				}
			}

			accumulate = (uint32_t)round((float)accumulate * inv_num_px_per_block);
			out[offset_out + out_col] = (uint8_t)accumulate;
		}
	}
}

uint64_t DHashNaive(const uint8_t resized_9x8[72])
{
	union
	{
		uint8_t _rows[8];
		uint64_t _hash;
	} hash;
	hash._hash = 0;

	for (int row = 1; row < 9; ++row)
	{
		for (int col = 0; col < 8; ++col)
		{
			uint64_t x = (resized_9x8[(row - 1) * 8 + col] > resized_9x8[row * 8 + col]);
			hash._hash |= (x << ((row-1)*8 + col));
		}
	}

	return hash._hash;
}

uint64_t DHashNaive(const uint8_t* RESTRICT data, const int width, const int height)
{
	alignas(32) uint8_t resized_9x8[72];
	// Resize the input image to 8x8
	Resize_2(data, width, height, resized_9x8);
	// Copy the first row down to 9th row
	*(uint64_t*)&resized_9x8[64] = *(uint64_t*)&resized_9x8[0];

	// Calculate DHash value based on that image.
	return DHashNaive(resized_9x8);
}

inline uint8_t Pack8U8(uint8_t* a)
{
	return (uint8_t)_pext_u64(*((uint64_t*)a), 0x0101010101010101ULL);
}

uint64_t DHash(const uint8_t* RESTRICT data, int width, int height)
{
	alignas(32) uint8_t resized_9x8[72];
	// Resize down to 8x8
	Resize_2(data, width, height, resized_9x8);
	// Copy the first row down to 9th row
	*(uint64_t*)&resized_9x8[64] = *(uint64_t*)&resized_9x8[0];

	// Load 0 1 2 3 each row is 8x8
	__m256i rows0 = _mm256_load_si256((__m256i*)&resized_9x8[0]);
	// Load 1 2 3 4 each row is 8x8
	__m256i rows1 = _mm256_load_si256((__m256i*)&resized_9x8[8]);
	// Load 4 5 6 7 each row is 8x8
	__m256i rows2 = _mm256_load_si256((__m256i*)&resized_9x8[32]);
	// Load 5 6 7 8 each row is 8x8
	__m256i rows3 = _mm256_load_si256((__m256i*)&resized_9x8[40]);

	__m256i _res0 = _mm256_cmpgt_epi8(rows0, rows1);
	__m256i _res1 = _mm256_cmpgt_epi8(rows2, rows3);

	alignas(32) uint8_t res0[32];
	alignas(32) uint8_t res1[32];

	_mm256_store_si256((__m256i*)&res0[0], _res0);
	_mm256_store_si256((__m256i*)&res1[0], _res1);

	union {
		struct {
			uint8_t l0;
			uint8_t l1;
			uint8_t l2;
			uint8_t l3;
			uint8_t l4;
			uint8_t l5;
			uint8_t l6;
			uint8_t l7;
		};
		uint64_t u64;
	} dhash;

	dhash.l0 = Pack8U8(&res0[0]);
	dhash.l1 = Pack8U8(&res0[8]);
	dhash.l2 = Pack8U8(&res0[16]);
	dhash.l3 = Pack8U8(&res0[24]);
	dhash.l4 = Pack8U8(&res1[0]);
	dhash.l5 = Pack8U8(&res1[8]);
	dhash.l6 = Pack8U8(&res1[16]);
	dhash.l7 = Pack8U8(&res1[24]);

	return dhash.u64;
}

uint32_t HammingDistance(uint64_t a, uint64_t b)
{
	return (uint32_t)__popcnt64(a ^ b);
}
