#ifndef KRON_H
#define KRON_H

#include <mkl.h>
#include <iostream>
#include <time.h>


class kron
{
	friend void pri(MKL_Complex8 *v);
public:
	explicit kron(int d, int EElength, int AElength);
	~kron();
	void kronUI(int d1);
	void kronIUI(int d1, int d2);
	void kronIU(int d1);
	void ProductState(int Nqubit);
	void Haar();
	MKL_Complex8 *GetValues();
	MKL_Complex8 *Getv();
	MKL_Complex8 *Getvv();
	MKL_Complex8 *Getvvv();
	int *GetCol();
	int *GetPB();
	int *GetPE();
	float *GetSch();
	float *GetEE();
	float *GetAE();
	void schmidtDecomposition(int dim, int dimA);
	void EECal(int Nqubit, int rate, int Nsample, int sample, int layer);
	void AECal(int Nqubit, int rate, int Nsample);

private:
	int d1;
	int d2;
	MKL_Complex8 U[16];
	MKL_Complex8 *valuesPtr;
	int *colPtr;
	int *pBPtr;
	int *pEPtr;
	MKL_Complex8 *vPtr;
	MKL_Complex8 *vvPtr;
	MKL_Complex8 *vvvPtr;
	float *SchcoefficientPtr;
	MKL_Complex8 *densityMatrixPtr;
	float *EEntropyPtr;
	float *AEntropyPtr;
};
#endif

