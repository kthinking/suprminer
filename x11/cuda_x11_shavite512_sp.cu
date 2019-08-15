/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
	optimized by sp - 2018
*/
#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define INTENSIVE_GMF
#include "cuda_x11_aes_sp.cuh"
__constant__ uint32_t c_PaddedMessage80[20]; // padded message (80 bytes + padding)

#define AESx(x) (x ##UL) /* SPH_C32(x) */

__device__ static uint32_t d_AES1[256] = {
	AESx(0x6363C6A5), AESx(0x7C7CF884), AESx(0x7777EE99), AESx(0x7B7BF68D),
	AESx(0xF2F2FF0D), AESx(0x6B6BD6BD), AESx(0x6F6FDEB1), AESx(0xC5C59154),
	AESx(0x30306050), AESx(0x01010203), AESx(0x6767CEA9), AESx(0x2B2B567D),
	AESx(0xFEFEE719), AESx(0xD7D7B562), AESx(0xABAB4DE6), AESx(0x7676EC9A),
	AESx(0xCACA8F45), AESx(0x82821F9D), AESx(0xC9C98940), AESx(0x7D7DFA87),
	AESx(0xFAFAEF15), AESx(0x5959B2EB), AESx(0x47478EC9), AESx(0xF0F0FB0B),
	AESx(0xADAD41EC), AESx(0xD4D4B367), AESx(0xA2A25FFD), AESx(0xAFAF45EA),
	AESx(0x9C9C23BF), AESx(0xA4A453F7), AESx(0x7272E496), AESx(0xC0C09B5B),
	AESx(0xB7B775C2), AESx(0xFDFDE11C), AESx(0x93933DAE), AESx(0x26264C6A),
	AESx(0x36366C5A), AESx(0x3F3F7E41), AESx(0xF7F7F502), AESx(0xCCCC834F),
	AESx(0x3434685C), AESx(0xA5A551F4), AESx(0xE5E5D134), AESx(0xF1F1F908),
	AESx(0x7171E293), AESx(0xD8D8AB73), AESx(0x31316253), AESx(0x15152A3F),
	AESx(0x0404080C), AESx(0xC7C79552), AESx(0x23234665), AESx(0xC3C39D5E),
	AESx(0x18183028), AESx(0x969637A1), AESx(0x05050A0F), AESx(0x9A9A2FB5),
	AESx(0x07070E09), AESx(0x12122436), AESx(0x80801B9B), AESx(0xE2E2DF3D),
	AESx(0xEBEBCD26), AESx(0x27274E69), AESx(0xB2B27FCD), AESx(0x7575EA9F),
	AESx(0x0909121B), AESx(0x83831D9E), AESx(0x2C2C5874), AESx(0x1A1A342E),
	AESx(0x1B1B362D), AESx(0x6E6EDCB2), AESx(0x5A5AB4EE), AESx(0xA0A05BFB),
	AESx(0x5252A4F6), AESx(0x3B3B764D), AESx(0xD6D6B761), AESx(0xB3B37DCE),
	AESx(0x2929527B), AESx(0xE3E3DD3E), AESx(0x2F2F5E71), AESx(0x84841397),
	AESx(0x5353A6F5), AESx(0xD1D1B968), AESx(0x00000000), AESx(0xEDEDC12C),
	AESx(0x20204060), AESx(0xFCFCE31F), AESx(0xB1B179C8), AESx(0x5B5BB6ED),
	AESx(0x6A6AD4BE), AESx(0xCBCB8D46), AESx(0xBEBE67D9), AESx(0x3939724B),
	AESx(0x4A4A94DE), AESx(0x4C4C98D4), AESx(0x5858B0E8), AESx(0xCFCF854A),
	AESx(0xD0D0BB6B), AESx(0xEFEFC52A), AESx(0xAAAA4FE5), AESx(0xFBFBED16),
	AESx(0x434386C5), AESx(0x4D4D9AD7), AESx(0x33336655), AESx(0x85851194),
	AESx(0x45458ACF), AESx(0xF9F9E910), AESx(0x02020406), AESx(0x7F7FFE81),
	AESx(0x5050A0F0), AESx(0x3C3C7844), AESx(0x9F9F25BA), AESx(0xA8A84BE3),
	AESx(0x5151A2F3), AESx(0xA3A35DFE), AESx(0x404080C0), AESx(0x8F8F058A),
	AESx(0x92923FAD), AESx(0x9D9D21BC), AESx(0x38387048), AESx(0xF5F5F104),
	AESx(0xBCBC63DF), AESx(0xB6B677C1), AESx(0xDADAAF75), AESx(0x21214263),
	AESx(0x10102030), AESx(0xFFFFE51A), AESx(0xF3F3FD0E), AESx(0xD2D2BF6D),
	AESx(0xCDCD814C), AESx(0x0C0C1814), AESx(0x13132635), AESx(0xECECC32F),
	AESx(0x5F5FBEE1), AESx(0x979735A2), AESx(0x444488CC), AESx(0x17172E39),
	AESx(0xC4C49357), AESx(0xA7A755F2), AESx(0x7E7EFC82), AESx(0x3D3D7A47),
	AESx(0x6464C8AC), AESx(0x5D5DBAE7), AESx(0x1919322B), AESx(0x7373E695),
	AESx(0x6060C0A0), AESx(0x81811998), AESx(0x4F4F9ED1), AESx(0xDCDCA37F),
	AESx(0x22224466), AESx(0x2A2A547E), AESx(0x90903BAB), AESx(0x88880B83),
	AESx(0x46468CCA), AESx(0xEEEEC729), AESx(0xB8B86BD3), AESx(0x1414283C),
	AESx(0xDEDEA779), AESx(0x5E5EBCE2), AESx(0x0B0B161D), AESx(0xDBDBAD76),
	AESx(0xE0E0DB3B), AESx(0x32326456), AESx(0x3A3A744E), AESx(0x0A0A141E),
	AESx(0x494992DB), AESx(0x06060C0A), AESx(0x2424486C), AESx(0x5C5CB8E4),
	AESx(0xC2C29F5D), AESx(0xD3D3BD6E), AESx(0xACAC43EF), AESx(0x6262C4A6),
	AESx(0x919139A8), AESx(0x959531A4), AESx(0xE4E4D337), AESx(0x7979F28B),
	AESx(0xE7E7D532), AESx(0xC8C88B43), AESx(0x37376E59), AESx(0x6D6DDAB7),
	AESx(0x8D8D018C), AESx(0xD5D5B164), AESx(0x4E4E9CD2), AESx(0xA9A949E0),
	AESx(0x6C6CD8B4), AESx(0x5656ACFA), AESx(0xF4F4F307), AESx(0xEAEACF25),
	AESx(0x6565CAAF), AESx(0x7A7AF48E), AESx(0xAEAE47E9), AESx(0x08081018),
	AESx(0xBABA6FD5), AESx(0x7878F088), AESx(0x25254A6F), AESx(0x2E2E5C72),
	AESx(0x1C1C3824), AESx(0xA6A657F1), AESx(0xB4B473C7), AESx(0xC6C69751),
	AESx(0xE8E8CB23), AESx(0xDDDDA17C), AESx(0x7474E89C), AESx(0x1F1F3E21),
	AESx(0x4B4B96DD), AESx(0xBDBD61DC), AESx(0x8B8B0D86), AESx(0x8A8A0F85),
	AESx(0x7070E090), AESx(0x3E3E7C42), AESx(0xB5B571C4), AESx(0x6666CCAA),
	AESx(0x484890D8), AESx(0x03030605), AESx(0xF6F6F701), AESx(0x0E0E1C12),
	AESx(0x6161C2A3), AESx(0x35356A5F), AESx(0x5757AEF9), AESx(0xB9B969D0),
	AESx(0x86861791), AESx(0xC1C19958), AESx(0x1D1D3A27), AESx(0x9E9E27B9),
	AESx(0xE1E1D938), AESx(0xF8F8EB13), AESx(0x98982BB3), AESx(0x11112233),
	AESx(0x6969D2BB), AESx(0xD9D9A970), AESx(0x8E8E0789), AESx(0x949433A7),
	AESx(0x9B9B2DB6), AESx(0x1E1E3C22), AESx(0x87871592), AESx(0xE9E9C920),
	AESx(0xCECE8749), AESx(0x5555AAFF), AESx(0x28285078), AESx(0xDFDFA57A),
	AESx(0x8C8C038F), AESx(0xA1A159F8), AESx(0x89890980), AESx(0x0D0D1A17),
	AESx(0xBFBF65DA), AESx(0xE6E6D731), AESx(0x424284C6), AESx(0x6868D0B8),
	AESx(0x414182C3), AESx(0x999929B0), AESx(0x2D2D5A77), AESx(0x0F0F1E11),
	AESx(0xB0B07BCB), AESx(0x5454A8FC), AESx(0xBBBB6DD6), AESx(0x16162C3A)
};


#define TPB 128

#define AESx(x) (x ##UL) /* SPH_C32(x) */

__device__ __forceinline__ void aes_round_s(const uint32_t sharedMemory[4][256], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, const uint32_t k0, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3){

	y0 = __ldg(&d_AES0[__byte_perm(x0, 0, 0x4440)]);
	y3 = sharedMemory[1][__byte_perm(x0, 0, 0x4441)];
	y2 = sharedMemory[2][__byte_perm(x0, 0, 0x4442)];
	y1 = sharedMemory[3][__byte_perm(x0, 0, 0x4443)];

	y1 ^= __ldg(&d_AES0[__byte_perm(x1, 0, 0x4440)]);
	y0 ^= sharedMemory[1][__byte_perm(x1, 0, 0x4441)];
	y3 ^= sharedMemory[2][__byte_perm(x1, 0, 0x4442)];
	y2 ^= sharedMemory[3][__byte_perm(x1, 0, 0x4443)];

	y0 ^= k0;

	y2 ^= __ldg(&d_AES0[__byte_perm(x2, 0, 0x4440)]);
	y1 ^= sharedMemory[1][__byte_perm(x2, 0, 0x4441)];
	y0 ^= sharedMemory[2][__byte_perm(x2, 0, 0x4442)];
	y3 ^= __ldg(&d_AES3[__byte_perm(x2, 0, 0x4443)]);

	y3 ^= __ldg(&d_AES3[__byte_perm(x3, 0, 0x4440)]);
	y2 ^= sharedMemory[1][__byte_perm(x3, 0, 0x4441)];
	y1 ^= sharedMemory[2][__byte_perm(x3, 0, 0x4442)];
	y0 ^= sharedMemory[3][__byte_perm(x3, 0, 0x4443)];
}

__device__ __forceinline__ void aes_round_s(const uint32_t sharedMemory[4][256], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3){

	y0 = __ldg(&d_AES0[__byte_perm(x0, 0, 0x4440)]);
	y3 = sharedMemory[1][__byte_perm(x0, 0, 0x4441)];
	y2 = sharedMemory[2][__byte_perm(x0, 0, 0x4442)];
	y1 = sharedMemory[3][__byte_perm(x0, 0, 0x4443)];

	y1 ^= __ldg(&d_AES0[__byte_perm(x1, 0, 0x4440)]);
	y0 ^= sharedMemory[1][__byte_perm(x1, 0, 0x4441)];
	y3 ^= sharedMemory[2][__byte_perm(x1, 0, 0x4442)];
	y2 ^= __ldg(&d_AES3[__byte_perm(x1, 0, 0x4443)]);

	y2 ^= __ldg(&d_AES0[__byte_perm(x2, 0, 0x4440)]);
	y1 ^= sharedMemory[1][__byte_perm(x2, 0, 0x4441)];
	y0 ^= sharedMemory[2][__byte_perm(x2, 0, 0x4442)];
	y3 ^= sharedMemory[3][__byte_perm(x2, 0, 0x4443)];

	y3 ^= __ldg(&d_AES0[__byte_perm(x3, 0, 0x4440)]);
	y2 ^= sharedMemory[1][__byte_perm(x3, 0, 0x4441)];
	y1 ^= sharedMemory[2][__byte_perm(x3, 0, 0x4442)];
	y0 ^= sharedMemory[3][__byte_perm(x3, 0, 0x4443)];
}


__device__ __forceinline__ void AES_ROUND_NOKEY_s(const uint32_t sharedMemory[4][256], uint4* x){

	uint32_t y0, y1, y2, y3;
	aes_round_s(sharedMemory, x->x, x->y, x->z, x->w, y0, y1, y2, y3);

	x->x = y0;
	x->y = y1;
	x->z = y2;
	x->w = y3;
}

__device__ __forceinline__ void KEY_EXPAND_ELT_s(const uint32_t sharedMemory[4][256], uint32_t *k)
{

	uint32_t y0, y1, y2, y3;
	aes_round_s(sharedMemory, k[0], k[1], k[2], k[3], y0, y1, y2, y3);

	k[0] = y1;
	k[1] = y2;
	k[2] = y3;
	k[3] = y0;
}


__device__ __forceinline__
void aes_gpu_init128_s(uint32_t sharedMemory[4][256])
{
	/* each thread startup will fill 2 uint32 */
	uint2 temp = __ldg(&((uint2*)&d_AES1)[threadIdx.x]);

//	sharedMemory[0][(threadIdx.x << 1) + 0] = temp.x;
//	sharedMemory[0][(threadIdx.x << 1) + 1] = temp.y;
	sharedMemory[1][(threadIdx.x << 1) + 0] = (temp.x);
	sharedMemory[1][(threadIdx.x << 1) + 1] = (temp.y);
	sharedMemory[2][(threadIdx.x << 1) + 0] = ROL8(temp.x);
	sharedMemory[2][(threadIdx.x << 1) + 1] = ROL8(temp.y);
	sharedMemory[3][(threadIdx.x << 1) + 0] = ROL16(temp.x);
	sharedMemory[3][(threadIdx.x << 1) + 1] = ROL16(temp.y);
}


__device__ __forceinline__ void round_3_7_11_s(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
	KEY_EXPAND_ELT_s(sharedMemory, &r[ 0]);
	*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
	x = p[ 2] ^ *(uint4*)&r[ 0];
	KEY_EXPAND_ELT_s(sharedMemory, &r[ 4]);
	r[4] ^= r[0];
	r[5] ^= r[1];
	r[6] ^= r[2];
	r[7] ^= r[3];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x.x ^= r[4];
	x.y ^= r[5];
	x.z ^= r[6];
	x.w ^= r[7];
	KEY_EXPAND_ELT_s(sharedMemory, &r[ 8]);
	r[8] ^= r[4];
	r[9] ^= r[5];
	r[10]^= r[6];
	r[11]^= r[7];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x.x ^= r[8];
	x.y ^= r[9];
	x.z ^= r[10];
	x.w ^= r[11];
	KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
	r[12] ^= r[8];
	r[13] ^= r[9];
	r[14]^= r[10];
	r[15]^= r[11];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x.x ^= r[12];
	x.y ^= r[13];
	x.z ^= r[14];
	x.w ^= r[15];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[ 1].x ^= x.x;
	p[ 1].y ^= x.y;
	p[ 1].z ^= x.z;
	p[ 1].w ^= x.w;
	KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[ 0] ^ *(uint4*)&r[16];
	KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[20];
	KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[24];
	KEY_EXPAND_ELT_s(sharedMemory,&r[28]);
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[ 3] ^= x;
}

__device__ __forceinline__
void round_4_8_12_s(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
	*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
	x = p[ 1] ^ *(uint4*)&r[ 0];
	AES_ROUND_NOKEY_s(sharedMemory, &x);

	r[ 4] ^= r[29];	r[ 5] ^= r[30];
	r[ 6] ^= r[31];	r[ 7] ^= r[ 0];

	x ^= *(uint4*)&r[ 4];
	*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[ 8];
	*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[ 0] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
	x = p[ 3] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY_s(sharedMemory, &x);
	p[ 2] ^= x;
}

// GPU Hash
__global__ __launch_bounds__(TPB,5) /* 64 registers with 128,8 - 72 regs with 128,7 */
void x11_shavite512_gpu_hash_64_sp(const uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint32_t sharedMemory[4][256];

	aes_gpu_init128_s(sharedMemory);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint4 p[ 4];
	uint4 x;
	uint32_t r[32];

	// kopiere init-state
	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC,	0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47,	0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};
//	if (thread < threads)
//	{
		uint64_t *Hash = &g_hash[thread<<3];

		// fülle die Nachricht mit 64-byte (vorheriger Hash)
		*(uint2x4*)&r[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		__syncthreads();
		
		*(uint2x4*)&p[ 0] = *(uint2x4*)&state[ 0];
		*(uint2x4*)&p[ 2] = *(uint2x4*)&state[ 8];
		r[16] = 0x80; r[17] = 0; r[18] = 0; r[19] = 0;
		r[20] = 0; r[21] = 0; r[22] = 0; r[23] = 0;
		r[24] = 0; r[25] = 0; r[26] = 0; r[27] = 0x02000000;
		r[28] = 0; r[29] = 0; r[30] = 0; r[31] = 0x02000000;
		/* round 0 */
		x = p[ 1] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);


		*(uint2x4*)&r[8] = __ldg4((uint2x4*)&Hash[4]);
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 0] ^= x;

		x = p[ 3];
		x.x ^= 0x80;
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x.w ^= 0x02000000;
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		x.w ^= 0x02000000;
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 2]^= x;


		// 1
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 0]);
		*(uint4*)&r[ 0]^=*(uint4*)&r[28];
		r[ 0] ^= 0x200;
		r[3] = ~r[3];

		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 1] ^= x;
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);

		r[ 4] ^= r[29]; r[ 5] ^= r[30];
		r[ 6] ^= r[31]; r[ 7] ^= r[ 0];

		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
//		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		r[24] ^= r[17];
		r[25] ^= r[18];
		r[26] ^= r[19];
		r[27] ^= r[20];


		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);

		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11_s(sharedMemory,r,p,x);
		
		
		/* round 4, 8, 12 */
		round_4_8_12_s(sharedMemory,r,p,x);

		// 2
		KEY_EXPAND_ELT_s(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		r[ 7] ^= (~0x200);
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 1] ^= x;
	
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11_s(sharedMemory,r,p,x);

		/* round 4, 8, 12 */
		round_4_8_12_s(sharedMemory,r,p,x);

		// 3
		KEY_EXPAND_ELT_s(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x^=*(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
		*(uint4*)&r[24]^=*(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		r[30] ^= 0x200;
		r[31] = ~r[31];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 1] ^= x;

		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		x = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		x = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11_s(sharedMemory,r,p,x);
		
		/* round 4, 8, 12 */
		round_4_8_12_s(sharedMemory,r,p,x);

		/* round 13 */
		KEY_EXPAND_ELT_s(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		x = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		x ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		x ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 3] ^= x;
		KEY_EXPAND_ELT_s(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		r[25] ^= 0x200;
		r[27] = ~r[27];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		KEY_EXPAND_ELT_s(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_s(sharedMemory, &x);
		p[ 1] ^= x;

		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&state[ 0] ^ *(uint2x4*)&p[ 2];
		*(uint2x4*)&Hash[ 4] = *(uint2x4*)&state[ 8] ^ *(uint2x4*)&p[ 0];
//	}
}

__device__ __forceinline__ void round_3_7_11(const uint32_t sharedMemory[8*1024], uint32_t* r, uint4 *p, uint4 &x){
	KEY_EXPAND_ELT_32(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[2] ^ *(uint4*)&r[0];
	KEY_EXPAND_ELT_32(sharedMemory, &r[4]);
	r[4] ^= r[0];
	r[5] ^= r[1];
	r[6] ^= r[2];
	r[7] ^= r[3];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x.x ^= r[4];
	x.y ^= r[5];
	x.z ^= r[6];
	x.w ^= r[7];
	KEY_EXPAND_ELT_32(sharedMemory, &r[8]);
	r[8] ^= r[4];
	r[9] ^= r[5];
	r[10] ^= r[6];
	r[11] ^= r[7];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x.x ^= r[8];
	x.y ^= r[9];
	x.z ^= r[10];
	x.w ^= r[11];
	KEY_EXPAND_ELT_32(sharedMemory, &r[12]);
	r[12] ^= r[8];
	r[13] ^= r[9];
	r[14] ^= r[10];
	r[15] ^= r[11];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x.x ^= r[12];
	x.y ^= r[13];
	x.z ^= r[14];
	x.w ^= r[15];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	p[1].x ^= x.x;
	p[1].y ^= x.y;
	p[1].z ^= x.z;
	p[1].w ^= x.w;
	KEY_EXPAND_ELT_32(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[0] ^ *(uint4*)&r[16];
	KEY_EXPAND_ELT_32(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x ^= *(uint4*)&r[20];
	KEY_EXPAND_ELT_32(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x ^= *(uint4*)&r[24];
	KEY_EXPAND_ELT_32(sharedMemory, &r[28]);
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	p[3] ^= x;
}

__device__ __forceinline__ void round_4_8_12(const uint32_t sharedMemory[8*1024], uint32_t* r, uint4 *p, uint4 &x){
	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[1] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY_32(sharedMemory, &x);

	r[4] ^= r[29];	r[5] ^= r[30];
	r[6] ^= r[31];	r[7] ^= r[0];

	x ^= *(uint4*)&r[4];
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x ^= *(uint4*)&r[8];
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	p[0] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[3] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY_32(sharedMemory, &x);
	p[2] ^= x;
}


__global__ __launch_bounds__(384, 2)
void x11_shavite512_gpu_hash_64_sp_final(const uint32_t threads, uint64_t *g_hash, uint32_t* resNonce, const uint64_t target)
{
	__shared__ uint32_t sharedMemory[8 * 1024];

	if (threadIdx.x<256) aes_gpu_init256_32(sharedMemory);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint4 p[4];
	uint4 x;
	uint32_t r[32];

	// kopiere init-state
	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC, 0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47, 0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};
	if (thread < threads)
	{
		uint2 *Hash = (uint2 *)&g_hash[thread << 3];

		// fülle die Nachricht mit 64-byte (vorheriger Hash)
		*(uint2x4*)&r[0] = __ldg4((uint2x4*)&Hash[0]);
		*(uint2x4*)&r[8] = __ldg4((uint2x4*)&Hash[4]);
		__syncthreads();

		*(uint2x4*)&p[0] = *(uint2x4*)&state[0];
		*(uint2x4*)&p[2] = *(uint2x4*)&state[8];
		r[16] = 0x80; r[17] = 0; r[18] = 0; r[19] = 0;
		r[20] = 0; r[21] = 0; r[22] = 0; r[23] = 0;
		r[24] = 0; r[25] = 0; r[26] = 0; r[27] = 0x02000000;
		r[28] = 0; r[29] = 0; r[30] = 0; r[31] = 0x02000000;
		/* round 0 */
		x = p[1] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[0] ^= x;
		x = p[3];
		x.x ^= 0x80;

		AES_ROUND_NOKEY_32(sharedMemory, &x);

		AES_ROUND_NOKEY_32(sharedMemory, &x);

		x.w ^= 0x02000000;
		AES_ROUND_NOKEY_32(sharedMemory, &x);

		x.w ^= 0x02000000;
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[2] ^= x;
		// 1
		KEY_EXPAND_ELT_32(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		r[0] ^= 0x200;
		r[3] = ~r[3];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[3] ^= x;
		KEY_EXPAND_ELT_32(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[1] ^= x;
		*(uint4*)&r[0] ^= *(uint4*)&r[25];
		x = p[3] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);

		r[4] ^= r[29]; r[5] ^= r[30];
		r[6] ^= r[31]; r[7] ^= r[0];

		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[8] ^= *(uint4*)&r[1];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[9];
		x = p[1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);

		p[0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory, r, p, x);


		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory, r, p, x);

		// 2
		KEY_EXPAND_ELT_32(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		r[7] ^= (~0x200);
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[3] ^= x;
		KEY_EXPAND_ELT_32(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[1] ^= x;

		*(uint4*)&r[0] ^= *(uint4*)&r[25];
		x = p[3] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		r[4] ^= r[29];
		r[5] ^= r[30];
		r[6] ^= r[31];
		r[7] ^= r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[8] ^= *(uint4*)&r[1];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[9];
		x = p[1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory, r, p, x);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory, r, p, x);

		// 3
		KEY_EXPAND_ELT_32(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[3] ^= x;
		KEY_EXPAND_ELT_32(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		r[30] ^= 0x200;
		r[31] = ~r[31];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[1] ^= x;

		*(uint4*)&r[0] ^= *(uint4*)&r[25];
		x = p[3] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		r[4] ^= r[29];
		r[5] ^= r[30];
		r[6] ^= r[31];
		r[7] ^= r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[8] ^= *(uint4*)&r[1];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[12] ^= *(uint4*)&r[5];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[2] ^= x;
		*(uint4*)&r[16] ^= *(uint4*)&r[9];
		x = p[1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[0] ^= x;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory, r, p, x);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory, r, p, x);

		/* round 13 */
		KEY_EXPAND_ELT_32(sharedMemory, &r[0]);
		*(uint4*)&r[0] ^= *(uint4*)&r[28];
		x = p[0] ^ *(uint4*)&r[0];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[4]);
		*(uint4*)&r[4] ^= *(uint4*)&r[0];
		x ^= *(uint4*)&r[4];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[8]);
		*(uint4*)&r[8] ^= *(uint4*)&r[4];
		x ^= *(uint4*)&r[8];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[8];
		x ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[3] ^= x;
		/*		KEY_EXPAND_ELT_32(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		x = p[2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		x ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		r[25] ^= 0x200;
		r[27] ^= 0xFFFFFFFF;
		x ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		KEY_EXPAND_ELT_32(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		x ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY_32(sharedMemory, &x);
		p[1] ^= x;
		*/
		//Hash[3] = 
		uint64_t test = (((uint64_t *)state)[3] ^ devectorize(make_uint2(p[3].z, p[3].w)));

		if (test <= target)
		{
			const uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}


__host__
void x11_shavite512_cpu_hash_64_sp(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	// note: 128 threads minimum are required to init the shared memory array
	x11_shavite512_gpu_hash_64_sp<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

__host__
void x11_shavite512_cpu_hash_64_sp_final(int thr_id, uint32_t threads, uint32_t *d_hash, const uint64_t target, uint32_t* resNonce)
{
	dim3 grid((threads + 384 - 1) / 384);
	dim3 block(384);

	// note: 128 threads minimum are required to init the shared memory array
	x11_shavite512_gpu_hash_64_sp_final << <grid, block >> >(threads, (uint64_t*)d_hash,resNonce,target);
}