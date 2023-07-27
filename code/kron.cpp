#include "kron.h"
#include <mkl.h>
#include <iostream>
#include <cmath>
#include <time.h>

using namespace std;

kron::kron(int d, int EElength, int AElength)
	: d1(0), d2(0),
	valuesPtr(new MKL_Complex8[16 * d]),
	colPtr(new int[16 * d]),
	pBPtr(new int[4 * d]),
	pEPtr(new int[4 * d]),
	vPtr(new MKL_Complex8[4 * d]),
	vvPtr(new MKL_Complex8[4 * d]),
	vvvPtr(new MKL_Complex8[4 * d]),
	SchcoefficientPtr(new float[2 * d]),
	densityMatrixPtr(new MKL_Complex8[4 * d]),
	EEntropyPtr(new float[EElength]),
	AEntropyPtr(new float[AElength])
{
	for (int i = 0; i < 16 * d; i++)
		valuesPtr[i] = { 0,0 };
	for (int i = 0; i < 16 * d; i++)
		colPtr[i] = 0;
	for (int i = 0; i < 4 * d; i++)
		pBPtr[i] = 0;
	for (int i = 0; i < 4 * d; i++)
		pEPtr[i] = 0;
	for (int i = 0; i < 4 * d; i++)
		vPtr[i] = { 0,0 };
	for (int i = 0; i < 4 * d; i++)
		vvPtr[i] = { 0,0 };
	for (int i = 0; i < 4 * d; i++)
		vvvPtr[i] = { 0,0 };
	for (int i = 0; i < 2 * d; i++)
		SchcoefficientPtr[i] = 0;
	for (int i = 0; i < 4 * d; i++)
		densityMatrixPtr[i] = { 0 ,0 };
	for (int i = 0; i < EElength; i++)
		EEntropyPtr[i] = 0;
	for (int i = 0; i < AElength; i++)
		AEntropyPtr[i] = 0;
}
kron::~kron()
{
	delete[] valuesPtr;
	delete[] colPtr;
	delete[] pBPtr;
	delete[] pEPtr;
	delete[] vPtr;
	delete[] vvPtr;
	delete[] vvvPtr;
	delete[] SchcoefficientPtr;
	delete[] densityMatrixPtr;
	delete[] EEntropyPtr;
	delete[] AEntropyPtr;
}
void kron::kronUI(int d)
{
	for (int i = 0; i < d; i++)
	{
		valuesPtr[4 * i] = U[0];
		valuesPtr[4 * i + 1] = U[1];
		valuesPtr[4 * i + 2] = U[2];
		valuesPtr[4 * i + 3] = U[3];
	}
	for (int i = 0; i < d; i++)
	{
		valuesPtr[4 * i + 4 * d] = U[4];
		valuesPtr[4 * i + 4 * d + 1] = U[5];
		valuesPtr[4 * i + 4 * d + 2] = U[6];
		valuesPtr[4 * i + 4 * d + 3] = U[7];
	}
	for (int i = 0; i < d; i++)
	{
		valuesPtr[4 * i + 8 * d] = U[8];
		valuesPtr[4 * i + 8 * d + 1] = U[9];
		valuesPtr[4 * i + 8 * d + 2] = U[10];
		valuesPtr[4 * i + 8 * d + 3] = U[11];
	}
	for (int i = 0; i < d; i++)
	{
		valuesPtr[4 * i + 12 * d] = U[12];
		valuesPtr[4 * i + 12 * d + 1] = U[13];
		valuesPtr[4 * i + 12 * d + 2] = U[14];
		valuesPtr[4 * i + 12 * d + 3] = U[15];
	}
	///////////////////////////////
	for (int i = 0; i < d; i++)
	{
		colPtr[4 * i] = i;
		colPtr[4 * i + 1] = i + d;
		colPtr[4 * i + 2] = i + 2 * d;
		colPtr[4 * i + 3] = i + 3 * d;
	}
	for (int i = 0; i < d; i++)
	{
		colPtr[4 * i + 4 * d] = i;
		colPtr[4 * i + 4 * d + 1] = i + d;
		colPtr[4 * i + 4 * d + 2] = i + 2 * d;
		colPtr[4 * i + 4 * d + 3] = i + 3 * d;
	}
	for (int i = 0; i < d; i++)
	{
		colPtr[4 * i + 8 * d] = i;
		colPtr[4 * i + 8 * d + 1] = i + d;
		colPtr[4 * i + 8 * d + 2] = i + 2 * d;
		colPtr[4 * i + 8 * d + 3] = i + 3 * d;
	}
	for (int i = 0; i < d; i++)
	{
		colPtr[4 * i + 12 * d] = i;
		colPtr[4 * i + 12 * d + 1] = i + d;
		colPtr[4 * i + 12 * d + 2] = i + 2 * d;
		colPtr[4 * i + 12 * d + 3] = i + 3 * d;
	}
	////////////////////////////////
	for (int i = 0; i < 4 * d; i++)
	{
		pBPtr[i] = 4 * i;
	}
	for (int i = 0; i < 4 * d; i++)
	{
		pEPtr[i] = 4 * i + 4;
	}
}
void kron::kronIUI(int d1, int d2)
{
	/////////////////////////////////
	for (int k = 0; k < d1; k++)
	{
		for (int i = 0; i < d2; i++)
		{
			valuesPtr[4 * i + 16 * d2 * k] = U[0];
			valuesPtr[4 * i + 1 + 16 * d2 * k] = U[1];
			valuesPtr[4 * i + 2 + 16 * d2 * k] = U[2];
			valuesPtr[4 * i + 3 + 16 * d2 * k] = U[3];
		}
		for (int i = 0; i < d2; i++)
		{
			valuesPtr[4 * i + 4 * d2 + 16 * d2 * k] = U[4];
			valuesPtr[4 * i + 4 * d2 + 16 * d2 * k + 1] = U[5];
			valuesPtr[4 * i + 4 * d2 + 16 * d2 * k + 2] = U[6];
			valuesPtr[4 * i + 4 * d2 + 16 * d2 * k + 3] = U[7];
		}
		for (int i = 0; i < d2; i++)
		{
			valuesPtr[4 * i + 8 * d2 + 16 * d2 * k] = U[8];
			valuesPtr[4 * i + 8 * d2 + 16 * d2 * k + 1] = U[9];
			valuesPtr[4 * i + 8 * d2 + 16 * d2 * k + 2] = U[10];
			valuesPtr[4 * i + 8 * d2 + 16 * d2 * k + 3] = U[11];
		}
		for (int i = 0; i < d2; i++)
		{
			valuesPtr[4 * i + 12 * d2 + 16 * d2 * k] = U[12];
			valuesPtr[4 * i + 12 * d2 + 16 * d2 * k + 1] = U[13];
			valuesPtr[4 * i + 12 * d2 + 16 * d2 * k + 2] = U[14];
			valuesPtr[4 * i + 12 * d2 + 16 * d2 * k + 3] = U[15];
		}
	}
	//////////////////////////////////
	for (int k = 0; k < d1; k++)
	{
		for (int i = 0; i < d2; i++)
		{
			colPtr[4 * i + 16 * d2 * k] = i + 4 * d2 * k;
			colPtr[4 * i + 16 * d2 * k + 1] = i + d2 + 4 * d2 * k;
			colPtr[4 * i + 16 * d2 * k + 2] = i + 2 * d2 + 4 * d2 * k;
			colPtr[4 * i + 16 * d2 * k + 3] = i + 3 * d2 + 4 * d2 * k;
		}
		for (int i = 0; i < d2; i++)
		{
			colPtr[4 * i + 4 * d2 + 16 * d2 * k] = i + 4 * d2 * k;
			colPtr[4 * i + 4 * d2 + 16 * d2 * k + 1] = i + d2 + 4 * d2 * k;
			colPtr[4 * i + 4 * d2 + 16 * d2 * k + 2] = i + 2 * d2 + 4 * d2 * k;
			colPtr[4 * i + 4 * d2 + 16 * d2 * k + 3] = i + 3 * d2 + 4 * d2 * k;
		}
		for (int i = 0; i < d2; i++)
		{
			colPtr[4 * i + 8 * d2 + 16 * d2 * k] = i + 4 * d2 * k;
			colPtr[4 * i + 8 * d2 + 16 * d2 * k + 1] = i + d2 + 4 * d2 * k;
			colPtr[4 * i + 8 * d2 + 16 * d2 * k + 2] = i + 2 * d2 + 4 * d2 * k;
			colPtr[4 * i + 8 * d2 + 16 * d2 * k + 3] = i + 3 * d2 + 4 * d2 * k;
		}
		for (int i = 0; i < d2; i++)
		{
			colPtr[4 * i + 12 * d2 + 16 * d2 * k] = i + 4 * d2 * k;
			colPtr[4 * i + 12 * d2 + 16 * d2 * k + 1] = i + d2 + 4 * d2 * k;
			colPtr[4 * i + 12 * d2 + 16 * d2 * k + 2] = i + 2 * d2 + 4 * d2 * k;
			colPtr[4 * i + 12 * d2 + 16 * d2 * k + 3] = i + 3 * d2 + 4 * d2 * k;
		}
	}
	///////////////////////////
	for (int k = 0; k < d1; k++)
	{
		for (int i = 0; i < 4 * d2; i++)
		{
			pBPtr[i + 4 * d2 * k] = 4 * i + 16 * d2 * k;
		}
		for (int i = 0; i < 4 * d2; i++)
		{
			pEPtr[i + 4 * d2 * k] = 4 * i + 16 * d2 * k + 4;
		}
	}
}

void kron::kronIU(int d)
{
	for (int i = 0; i < d; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			valuesPtr[16 * i + j] = U[j];
		}
	}

	for (int i = 0; i < d; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				colPtr[16 * i + 4 * j + k] = k + 4 * i;
			}
		}
	}

	for (int i = 0; i < d; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			pBPtr[4 * i + j] = 4 * j + 16 * i;
		}
		for (int j = 0; j < 4; j++)
		{
			pEPtr[4 * i + j] = 4 * j + 16 * i + 4;
		}
	}

}

void kron::ProductState(int Nqubit)
{
	//unique_ptr<MKL_Complex8[]> vv(new MKL_Complex8[(int)pow(2, Nqubit)]);

	MKL_Complex8 v2[2];

	for (int i = 0; i < Nqubit; i++)
	{
		for (int j = 0; j < (int)pow(2, i + 1); j++)
		{

			if (i == 0)
			{
				vvPtr[j] = { (float)rand() / RAND_MAX,(float)rand() / RAND_MAX };

				if (j == 1)
				{
					float nvvPtr = sqrt(vvPtr[0].real*vvPtr[0].real + vvPtr[0].imag*vvPtr[0].imag
						+ vvPtr[1].real*vvPtr[1].real + vvPtr[1].imag*vvPtr[1].imag);
					vvPtr[0].real = vvPtr[0].real / nvvPtr;
					vvPtr[1].real = vvPtr[1].real / nvvPtr;
					vvPtr[0].imag = vvPtr[0].imag / nvvPtr;
					vvPtr[1].imag = vvPtr[1].imag / nvvPtr;


					for (int k = 0; k < (int)pow(2, i + 1); k++)
					{
						vPtr[k] = vvPtr[k];
					}
				}

				continue;
			}

			if (j == 0)
			{
				v2[0] = { (float)rand() / RAND_MAX,(float)rand() / RAND_MAX };
				v2[1] = { (float)rand() / RAND_MAX,(float)rand() / RAND_MAX };
				float nv2 = sqrt(v2[0].real*v2[0].real + v2[0].imag*v2[0].imag
					+ v2[1].real*v2[1].real + v2[1].imag*v2[1].imag);
				v2[0].real = v2[0].real / nv2;
				v2[1].real = v2[1].real / nv2;
				v2[0].imag = v2[0].imag / nv2;
				v2[1].imag = v2[1].imag / nv2;
			}


			int p1;
			if ((j + 1) % 2 == 0)
				p1 = (j + 1) / 2 - 1;
			else
				p1 = (j + 1) / 2 + 1 - 1;

			vPtr[j].real = vvPtr[p1].real * v2[j % 2].real - vvPtr[p1].imag * v2[j % 2].imag;
			vPtr[j].imag = vvPtr[p1].imag * v2[j % 2].real + vvPtr[p1].real * v2[j % 2].imag;

		}
		for (int k = 0; k < (int)pow(2, i + 1); k++)
		{
			vvPtr[k] = vPtr[k];
		}

	}
}
void kron::Haar()
{
	//float UR[16];
	//float UI[16];
	MKL_Complex8 tau[4] = {};
	MKL_Complex8 ttau[16] = {};
	MKL_Complex8 Q[16];
	MKL_Complex8 UU[16];
	MKL_Complex8 aph = { (float)1,0 };
	//MKL_Complex8 *aphPtr = &aph;
	MKL_Complex8 beta = { (float)0,0 };
	//MKL_Complex8 *betaPtr = &beta;
	//VSLStreamStatePtr stream;
	//int errcode = vslNewStream(&stream, VSL_BRNG_MCG31, rand());
	//vsRngGaussian(0, stream, 16, UR, 0, 1);
	//vsRngGaussian(0, stream, 16, UI, 0, 1);
	for (int i = 0; i < 16; i++)
	{
		U[i] = { rand() / (float)RAND_MAX * (float)sqrt(2.0) ,rand() / (float)RAND_MAX * (float)sqrt(2.0) };
	}

	LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, 4, 4, U, 4, tau);
	for (int i = 0; i < 4; i++)
	{
		ttau[5 * i].real = U[5 * i].real / sqrt(U[5 * i].real*U[5 * i].real + U[5 * i].imag*U[5 * i].imag);
		ttau[5 * i].imag = U[5 * i].imag / sqrt(U[5 * i].real*U[5 * i].real + U[5 * i].imag*U[5 * i].imag);
	}
	LAPACKE_cungqr(LAPACK_ROW_MAJOR, 4, 4, 4, U, 4, tau);

	for (int i = 0; i < 16; i++)
	{
		Q[i] = U[i];
	}

	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, &aph, Q, 4, ttau, 4, &beta, U, 4);
	//cblas_cgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, &aph, UU, 4, UU, 4, &beta, U, 4);
}

MKL_Complex8 *kron::GetValues()
{
	return valuesPtr;
}
MKL_Complex8 *kron::Getv()
{
	return vPtr;
}
MKL_Complex8 *kron::Getvv()
{
	return vvPtr;
}
MKL_Complex8 *kron::Getvvv()
{
	return vvvPtr;
}
int *kron::GetCol()
{
	return colPtr;
}
int *kron::GetPB()
{
	return pBPtr;
}
int *kron::GetPE()
{
	return pEPtr;
}
float *kron::GetSch()
{
	return SchcoefficientPtr;
}
float *kron::GetEE()
{
	return EEntropyPtr;
}
float *kron::GetAE()
{
	return AEntropyPtr;
}
void kron::schmidtDecomposition(int dim, int dimA )
{
	//unique_ptr<MKL_Complex8[]> densityMatrix(new MKL_Complex8[dimA*dimA]);
	//MKL_Complex8 *densityMatrix = new MKL_Complex8[dimA*dimA];
	MKL_Complex8 u[dimA*dimA];
	MKL_Complex8 vt[dimA*dimA];

	for (int i = 0; i < dimA; i++)
	{
		for (int j = 0; j < dimA; j++)
		{
			densityMatrixPtr[dimA*i + j] = vPtr[(dimA*i + j)];
		}
	}
	MKL_INT info;
	info = LAPACKE_cgesdd(LAPACK_ROW_MAJOR, 'S', dimA, dimA, densityMatrixPtr, dimA, SchcoefficientPtr, u, dimA, vt, dimA);
	// cout << "info : " << info << endl;

// LAPACKE_cgesdd( LAPACK_ROW_MAJOR, 'S', m, n, a, lda, s, u, ldu, vt, ldvt );
// 	LAPACKE_cgesdd(LAPACK_ROW_MAJOR, 'N', dimA, dimA, densityMatrixPtr, dimA, SchcoefficientPtr, u, 1, vt, 1);
}
void kron::EECal(int Nqubit, int rate, int Nsample, int sample, int layer)
{
	for (int i = 0; i < (int)pow(2, Nqubit / 2); i++)
	{
		if (SchcoefficientPtr[i] < pow(10, -5))
		{
			break;
		}
		EEntropyPtr[rate*Nsample*(Nqubit + 1) + sample * (Nqubit + 1) + layer] = EEntropyPtr[rate*Nsample*(Nqubit + 1) + sample * (Nqubit + 1) + layer]
			- SchcoefficientPtr[i] * SchcoefficientPtr[i] * log2(SchcoefficientPtr[i] * SchcoefficientPtr[i]);
	}
}
void kron::AECal(int Nqubit, int rate, int Nsample)
{
	for (int i = 1; i < Nqubit + 1; i++)
	{
		for (int sample = 0; sample < Nsample; sample++)
		{
			AEntropyPtr[rate*(Nqubit + 1) + i] =
				AEntropyPtr[rate*(Nqubit + 1) + i]
				+ EEntropyPtr[rate*Nsample*(Nqubit + 1) + sample * (Nqubit + 1) + i];
		}
		AEntropyPtr[rate*(Nqubit + 1) + i] = AEntropyPtr[rate*(Nqubit + 1) + i] / Nsample;
	}
}
