#include "DHash.h"
#include <iostream>
#include <bitset>

constexpr int DHASH_WIDTH = 8;
constexpr int DHASH_HEIGHT = 8;
constexpr int ALLOC_ALIGN = 32;

int TestResize()
{
	srand((unsigned int)time(NULL));

	constexpr int WIDTH = 2133;
	constexpr int HEIGHT = 2133;
	constexpr int NUM_TESTS_INNER = 1;
	constexpr int NUM_TESTS_OUTER = 100;

	uint8_t* image = (uint8_t*)_aligned_malloc(WIDTH * HEIGHT, ALLOC_ALIGN);
	if (!image) return 1;
	// Initialize the image with random values
	{
		memset(image, 0, WIDTH * HEIGHT);
		for (int idx = 0; idx < WIDTH * HEIGHT; ++idx)
		{
			image[idx] = (uint8_t)(
					((float)rand() / (RAND_MAX + 1.0f)) * 100.0f +
					((float)idx / WIDTH * HEIGHT) * 155.0f
				);
		}
	}

	uint8_t* out_buf;
	uint8_t* out_image_1;
	uint8_t* out_image_2;

	{
		size_t out_image_size = DHASH_HEIGHT * DHASH_WIDTH;
		size_t buf_size = 2 * out_image_size;
		out_buf = (uint8_t*)malloc(buf_size);
		if (!out_buf)
			return 1;
		memset(out_buf, 0, buf_size);
		out_image_1 = &out_buf[0];
		out_image_2 = &out_buf[out_image_size];
	}

	double resize_naive_t = 0.0;
	double resize_1_t = 0.0;
	double resize_2_t = 0.0;
	double resize_3_t = 0.0;

	// Write to the first output at least once, so we can disable the loop inside without giving up the asserts.
	{
		for (int t = 0; t < NUM_TESTS_INNER; ++t)
		{
			Resize_Naive(image, WIDTH, HEIGHT, out_image_1);
		}
	}

	for (int test = 0; test < NUM_TESTS_OUTER; ++test)
	{
		{
			const auto t1 = std::chrono::high_resolution_clock::now();
			for (int t = 0; t < NUM_TESTS_INNER; ++t)
			{
				Resize_Naive(image, WIDTH, HEIGHT, out_image_1);
			}
			const auto t2 = std::chrono::high_resolution_clock::now();

			int64_t time_passed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
			resize_naive_t += (double)time_passed / NUM_TESTS_INNER;
		}

		{
			const auto t1 = std::chrono::high_resolution_clock::now();
			for (int t = 0; t < NUM_TESTS_INNER; ++t)
			{
				Resize_1(image, WIDTH, HEIGHT, out_image_2);
			}
			const auto t2 = std::chrono::high_resolution_clock::now();

			int64_t time_passed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
			resize_1_t += (double)time_passed / NUM_TESTS_INNER;

			// Assert that the images are the same.
			for (int i = 0, _e = DHASH_WIDTH * DHASH_HEIGHT; i < _e; ++i)
			{
				ASSERT(out_image_1[i] == out_image_2[i]);
			}
		}

		{
			const auto t1 = std::chrono::high_resolution_clock::now();
			for (int t = 0; t < NUM_TESTS_INNER; ++t)
			{
				Resize_2(image, WIDTH, HEIGHT, out_image_2);
			}
			const auto t2 = std::chrono::high_resolution_clock::now();

			int64_t time_passed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
			resize_2_t += (double)time_passed / NUM_TESTS_INNER;

			// Assert that the images are the same.
			for (int i = 0, _e = DHASH_WIDTH * DHASH_HEIGHT; i < _e; ++i)
			{
				ASSERT(out_image_1[i] == out_image_2[i]);
			}
		}

		{
			const auto t1 = std::chrono::high_resolution_clock::now();
			for (int t = 0; t < NUM_TESTS_INNER; ++t)
			{
				Resize_3(image, WIDTH, HEIGHT, out_image_2);
			}
			const auto t2 = std::chrono::high_resolution_clock::now();

			int64_t time_passed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
			resize_3_t += (double)time_passed / NUM_TESTS_INNER;

			// Assert that the images are the same.
			for (int i = 0, _e = DHASH_WIDTH * DHASH_HEIGHT; i < _e; ++i)
			{
				ASSERT(out_image_1[i] == out_image_2[i]);
			}
		}
	}

	resize_naive_t /= NUM_TESTS_OUTER;
	resize_1_t /= NUM_TESTS_OUTER;
	resize_2_t /= NUM_TESTS_OUTER;
	resize_3_t /= NUM_TESTS_OUTER;

	printf("Avg time for naive impl: %f\n", resize_naive_t);
	printf("Avg time for 1st simd impl: %f\n", resize_1_t);
	printf("Avg time for 2nd simd impl: %f\n", resize_2_t);
	printf("Avg time for 3nd simd impl: %f\n", resize_3_t);

	_aligned_free(image);
	free(out_buf);

	return 0;
}

int TestDHash()
{
	srand((unsigned int)time(NULL));

	constexpr int WIDTH = 2133;
	constexpr int HEIGHT = 2133;

	uint8_t* image = (uint8_t*)_aligned_malloc(WIDTH * HEIGHT, ALLOC_ALIGN);
	if (!image) return 1;
	// Initialize the image with random values
	{
		memset(image, 0, WIDTH * HEIGHT);
		for (int idx = 0; idx < WIDTH * HEIGHT; ++idx)
		{
			image[idx] = (uint8_t)(
					((float)rand() / (RAND_MAX + 1.0f)) * 100.0f +
					((float)idx / WIDTH * HEIGHT) * 155.0f
				);
		}
	}

	uint64_t dhash_1 = DHashNaive(image, WIDTH, HEIGHT);
	uint64_t dhash_2 = DHash(image, WIDTH, HEIGHT);
	uint32_t diff = HammingDistance(dhash_1, dhash_2);

	ASSERT(diff == 0);

	_aligned_free(image);

	return 0;
}

int main(int argc, char* argv[])
{
	(void)argc;
	(void)argv;

	TestDHash();
	TestResize();
}
