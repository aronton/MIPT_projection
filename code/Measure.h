#ifndef MEASURE_H
#define MEASURE_H

#include <mkl.h>

class Measure
{
public:
	explicit Measure(int d);
	~Measure();
	void measurePI_plus(int d2);
	void measurePI_minus(int d2);
	void measureIPI_plus(int d1, int d2);
	void measureIPI_minus(int d1, int d2);
	void measureIP_plus(int d1);
	void measureIP_minus(int d1);
	MKL_Complex8 *GetValues();
	int *GetCol();
	int *GetPB();
	int *GetPE();
private:
	int d1;
	int d2;
	MKL_Complex8 *valuesPtr;
	int *colPtr;
	int *pBPtr;
	int *pEPtr;
};
#endif


