/**
* X16R algorithm (X16 with Randomized chain order)
*
* tpruvot 2018 - GPL code
*/

#include <stdio.h>
#include <memory.h>
#include <unistd.h>

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x16r.h"

extern void quark_bmw512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t *resNonce, const uint64_t target);
extern void x11_luffa512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint64_t target, uint32_t *d_resNonce);
extern void tribus_echo512_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void x16_simd_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_cubehash_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_outputHash, uint32_t *resNonce, const uint64_t target);
extern void x13_fugue512_cpu_hash_64_final_sp(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void x16_simd_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x16_simd_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x16_simd_whirlpool512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);


static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t h_resNonce[MAX_GPUS][4];


static uint32_t *d_hash[MAX_GPUS];

enum Algo {
	BLAKE = 0,
	BMW,
	GROESTL,
	JH,
	KECCAK,
	SKEIN,
	LUFFA,
	CUBEHASH,
	SHAVITE,
	SIMD,
	ECHO,
	HAMSI,
	FUGUE,
	SHABAL,
	WHIRLPOOL,
	SHA512,
	HASH_FUNC_COUNT
};

static const char* algo_strings[] = {
	"blake",
	"bmw512",
	"groestl",
	"jh512",
	"keccak",
	"skein",
	"luffa",
	"cube",
	"shavite",
	"simd",
	"echo",
	"hamsi",
	"fugue",
	"shabal",
	"whirlpool",
	"sha512",
	NULL
};

static __thread uint32_t s_ntime = UINT32_MAX;
static __thread char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };

static void getAlgoString(const uint32_t* prevblock, char *output)
{

	uint8_t* data = (uint8_t*)prevblock;

	strcpy(output, "0123456789ABCDEF");

	for (int i = 0; i < 16; i++){
		uint8_t b = (15 - i) >> 1; // 16 ascii hex chars, reversed
		uint8_t algoDigit = (i & 1) ? data[b] & 0xF : data[b] >> 4;
		int offset = algoDigit;
		// insert the nth character at the front
		char oldVal = output[offset];
		for (int j = offset; j-->0;){
			output[j + 1] = output[j];
		}
		output[0] = oldVal;
	}
}


// X16R CPU Hash (Validation)
extern "C" void x16s_hash(void *output, const void *input)
{
	unsigned char _ALIGN(64) hash[128];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_fugue512_context ctx_fugue;
	sph_shabal512_context ctx_shabal;
	sph_whirlpool_context ctx_whirlpool;
	sph_sha512_context ctx_sha512;

	void *in = (void*)input;
	int size = 80;

	uint32_t *in32 = (uint32_t*)input;
	getAlgoString(&in32[1], hashOrder);

	for (int i = 0; i < 16; i++)
	{
		const char elem = hashOrder[i];
		const uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

		switch (algo) {
		case BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, in, size);
			sph_blake512_close(&ctx_blake, hash);
			break;
		case BMW:
			sph_bmw512_init(&ctx_bmw);
			sph_bmw512(&ctx_bmw, in, size);
			sph_bmw512_close(&ctx_bmw, hash);
			break;
		case GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, in, size);
			sph_groestl512_close(&ctx_groestl, hash);
			break;
		case SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, in, size);
			sph_skein512_close(&ctx_skein, hash);
			break;
		case JH:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, in, size);
			sph_jh512_close(&ctx_jh, hash);
			break;
		case KECCAK:
			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, in, size);
			sph_keccak512_close(&ctx_keccak, hash);
			break;
		case LUFFA:
			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, in, size);
			sph_luffa512_close(&ctx_luffa, hash);
			break;
		case CUBEHASH:
			sph_cubehash512_init(&ctx_cubehash);
			sph_cubehash512(&ctx_cubehash, in, size);
			sph_cubehash512_close(&ctx_cubehash, hash);
			break;
		case SHAVITE:
			sph_shavite512_init(&ctx_shavite);
			sph_shavite512(&ctx_shavite, in, size);
			sph_shavite512_close(&ctx_shavite, hash);
			break;
		case SIMD:
			sph_simd512_init(&ctx_simd);
			sph_simd512(&ctx_simd, in, size);
			sph_simd512_close(&ctx_simd, hash);
			break;
		case ECHO:
			sph_echo512_init(&ctx_echo);
			sph_echo512(&ctx_echo, in, size);
			sph_echo512_close(&ctx_echo, hash);
			break;
		case HAMSI:
			sph_hamsi512_init(&ctx_hamsi);
			sph_hamsi512(&ctx_hamsi, in, size);
			sph_hamsi512_close(&ctx_hamsi, hash);
			break;
		case FUGUE:
			sph_fugue512_init(&ctx_fugue);
			sph_fugue512(&ctx_fugue, in, size);
			sph_fugue512_close(&ctx_fugue, hash);
			break;
		case SHABAL:
			sph_shabal512_init(&ctx_shabal);
			sph_shabal512(&ctx_shabal, in, size);
			sph_shabal512_close(&ctx_shabal, hash);
			break;
		case WHIRLPOOL:
			sph_whirlpool_init(&ctx_whirlpool);
			sph_whirlpool(&ctx_whirlpool, in, size);
			sph_whirlpool_close(&ctx_whirlpool, hash);
			break;
		case SHA512:
			sph_sha512_init(&ctx_sha512);
			sph_sha512(&ctx_sha512, (const void*)in, size);
			sph_sha512_close(&ctx_sha512, (void*)hash);
			break;
		}
		in = (void*)hash;
		size = 64;
	}
	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

//#define _DEBUG
#define _DEBUG_PREFIX "x16s-"
#include "cuda_debug.cuh"

//static int algo80_tests[HASH_FUNC_COUNT] = { 0 };
//static int algo64_tests[HASH_FUNC_COUNT] = { 0 };
static int algo80_fails[HASH_FUNC_COUNT] = { 0 };

extern "C" int scanhash_x16s(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500) ? 20 : 19;
	if (strstr(device_name[dev_id], "GTX 1080")) intensity = 21;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);
	throughput &= 0xFFFFFF00; //multiples of 128 due to cubehash_shavite & simd_echo kernels

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput); // 64
		x16_echo512_cuda_init(thr_id, throughput);
		x16_fugue512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x16_whirlpool512_init(thr_id, throughput);

		x11_luffa512_cpu_init(thr_id, throughput); // 64
		x16_echo512_cuda_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x16_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t)64 * throughput), 0);
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));

		cuda_check_cpu_init(thr_id, throughput);
		sleep(2);
		init[thr_id] = true;
	}

/*	if (opt_benchmark) 
	{
		((uint32_t*)ptarget)[7] = 0x03f;
		((uint32_t*)pdata)[1] = 0x01234567;
		((uint32_t*)pdata)[2] = 0x89ABCDEF;
		//((uint8_t*)pdata)[8] = 0x90; // hashOrder[0] = '9'; for simd 80 + blake512 64
		//((uint8_t*)pdata)[8] = 0xA0; // hashOrder[0] = 'A'; for echo 80 + blake512 64
		//((uint8_t*)pdata)[8] = 0xB0; // hashOrder[0] = 'B'; for hamsi 80 + blake512 64
		//((uint8_t*)pdata)[8] = 0xC0; // hashOrder[0] = 'C'; for fugue 80 + blake512 64
		//((uint8_t*)pdata)[8] = 0xE0; // hashOrder[0] = 'E'; for whirlpool 80 + blake512 64
	}
*/

	uint32_t  endiandata[20];

	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	uint32_t ntime = swab32(pdata[17]);
	if (s_ntime != ntime) 
	{
		getAlgoString(&endiandata[1], hashOrder);
		s_ntime = ntime;

		char *temp = "C9C9C9C9C9C9C9C9";
		if (opt_benchmark)
		{
			for (int k = 0; k < 16; k++)
				hashOrder[k] = temp[k];
		}
		if (!thr_id) applog(LOG_INFO, "hash order %s (%08x)", hashOrder, ntime);
	}



	cuda_check_cpu_setTarget(ptarget);

	char elem = hashOrder[0];
	const uint8_t algo80 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

	switch (algo80) {
	case BLAKE:
		quark_blake512_cpu_setBlock_80(thr_id, endiandata);
		break;
	case BMW:
		quark_bmw512_cpu_setBlock_80(endiandata);
		break;
	case GROESTL:
		groestl512_setBlock_80(thr_id, endiandata);
		break;
	case JH:
		jh512_setBlock_80(thr_id, endiandata);
		break;
	case KECCAK:
		keccak512_setBlock_80(thr_id, endiandata);
		break;
	case SKEIN:
		skein512_cpu_setBlock_80((void*)endiandata);
		break;
	case LUFFA:
		qubit_luffa512_cpu_setBlock_80_alexis((void*)endiandata);
		break;
	case CUBEHASH:
		cubehash512_setBlock_80(thr_id, endiandata);
		break;
	case SHAVITE:
		x11_shavite512_setBlock_80((void*)endiandata);
		break;
	case SIMD:
		x16_simd512_setBlock_80((void*)endiandata);
		break;
	case ECHO:
		x16_echo512_setBlock_80((void*)endiandata);
		break;
	case HAMSI:
		x16_hamsi512_setBlock_80((uint64_t*)endiandata);
		break;
	case FUGUE:
		x16_fugue512_setBlock_80((void*)pdata);
		break;
	case SHABAL:
		x16_shabal512_setBlock_80((void*)endiandata);
		break;
	case WHIRLPOOL:
		x16_whirlpool512_setBlock_80((void*)endiandata);
		break;
	case SHA512:
		x16_sha512_setBlock_80(endiandata);
		break;
	default: {
		if (!thr_id)
			applog(LOG_WARNING, "kernel %s %c unimplemented, order %s", algo_strings[algo80], elem, hashOrder);
		sleep(5);
		return -1;
	}
	}

	int warn = 0;

	do {

		int order = 0;
		bool addstart = false;

		// Hash with CUDA

		switch (algo80) {
		case BLAKE:
			quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case BMW:
			quark_bmw512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
			break;
		case GROESTL:
			groestl512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case JH:
			jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case KECCAK:
			keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case SKEIN:
			skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1); order++;
			break;
		case LUFFA:
			qubit_luffa512_cpu_hash_80_alexis(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case CUBEHASH:
			cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case SHAVITE:
			x11_shavite512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
			break;
		case SIMD:
			x16_simd512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case ECHO:
			x16_echo512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case HAMSI:
			x16_hamsi512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case FUGUE:
			x16_fugue512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case SHABAL:
			x16_shabal512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case WHIRLPOOL:
			x16_whirlpool512_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		case SHA512:
			x16_sha512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
			break;
		}

		for (int i = 1; i < 16; i++)
		{
			const char elem = hashOrder[i];
			const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

			uint8_t nextalgo = 50;
			if (i < 15)
			{
				const char elem2 = hashOrder[i + 1];
				nextalgo = elem2 >= 'A' ? elem2 - 'A' + 10 : elem2 - '0';
			}

			switch (algo64) 
			{
			case BLAKE:
				if (i == 15)
				{
					quark_blake512_cpu_hash_64_final(thr_id, throughput, NULL, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{

					quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				}

				break;
			case BMW:
				if (i == 15)
				{
					quark_bmw512_cpu_hash_64_final(thr_id, throughput, NULL, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				}

				break;
			case GROESTL:
				quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				break;
			case JH:
				quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				break;
			case KECCAK:
				//quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
				break;
			case SKEIN:
				quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				break;
			case LUFFA:
				if (i == 15)
				{
					x11_luffa512_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], ((uint64_t *)ptarget)[3], d_resNonce[thr_id]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x11_luffa512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				}
				break;
			case CUBEHASH:
				if (nextalgo == SHAVITE)
				{
					x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
					i = i + 1;
				}
				else
				{
					x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
				}
				break;
			case SHAVITE:
				x11_shavite512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				break;
			case SIMD:
				if (nextalgo == ECHO)
				{
					x16_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
					i = i + 1;
				} 
				else if (nextalgo == WHIRLPOOL)
				{
					x16_simd_whirlpool512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
					i = i + 1;
				}
				else if (nextalgo == HAMSI)
				{
					x16_simd_hamsi512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
					i = i + 1;
				}
				else if (nextalgo == FUGUE)
				{
					x16_simd_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
					i = i + 1;
				}
				else
				{
					x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				}
				break;
			case ECHO:
				if (i == 15)
				{
					tribus_echo512_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x11_echo512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				}
				break;
			case HAMSI:
				x13_hamsi512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				break;
			case FUGUE:
				if (i == 15)
				{
					x13_fugue512_cpu_hash_64_final_sp(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], ((uint64_t *)ptarget)[3]);
					CUDA_SAFE_CALL(cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
					work->nonces[0] = h_resNonce[thr_id][0];
					addstart = true;
				}
				else
				{
					x13_fugue512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]);
				}
				break;
			case SHABAL:
				x14_shabal512_cpu_hash_64_alexis(thr_id, throughput, d_hash[thr_id]); order++;
				break;
			case WHIRLPOOL:
				x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				break;
			case SHA512:
				x17_sha512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
				break;
			}
		}

		*hashes_done = pdata[19] - first_nonce + throughput;
		if (!addstart)
		{
			work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		}

		if (work->nonces[0] != UINT32_MAX)
		{
			if (opt_benchmark) gpulog(LOG_BLUE, dev_id, "found");

			if (addstart) work->nonces[0] += pdata[19];
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			x16s_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) 
			{
				work->valid_nonces = 1;
				int res = 1;
				//work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				
				if (addstart && (h_resNonce[thr_id][1] != UINT32_MAX))
				{
					work->nonces[1] = h_resNonce[thr_id][1] + pdata[19];
				}
				if (!addstart)
				{
					work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
					if (work->nonces[1] == 0) work->nonces[1] = UINT32_MAX;
				}
				
				
				work_set_target_ratio(work, vhash);
				*hashes_done = pdata[19] - first_nonce + throughput;
				pdata[19] = work->nonces[0];
				if (work->nonces[1] != UINT32_MAX)
				{
					//					gpulog(LOG_BLUE, dev_id, "found2");
					//					if(!opt_quiet)
					//						gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08x", secNonce);
					be32enc(&endiandata[19], work->nonces[1]);
					pdata[21] = work->nonces[1];
					x16s_hash(vhash, endiandata);
					if (bn_hash_target_ratio(vhash, ptarget) > work->shareratio[0]){
						work_set_target_ratio(work, vhash);
						xchg(pdata[19], pdata[21]);
					}
					res++;
					work->valid_nonces = 2;
				}
				return res;
			}
			else if (vhash[7] > Htarg) {
				// x11+ coins could do some random error, but not on retry
				gpu_increment_reject(thr_id);
				algo80_fails[algo80]++;
				if (!warn) {
					warn++;
					pdata[19] = work->nonces[0] + 1;
					continue;
				}
				else {
					if (!opt_quiet)	gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! %s %s",
						work->nonces[0], algo_strings[algo80], hashOrder);
					warn = 0;
				}
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_x16s(int thr_id)
{
	if (!init[thr_id])
		return;

	 cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);
	x13_fugue512_cpu_free(thr_id);
	x16_fugue512_cpu_free(thr_id); // to merge with x13_fugue512 ?
	x15_whirlpool_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}