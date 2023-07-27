#include "Measure.h"
#include <mkl.h>
#include <iostream>
#include <cmath>

using namespace std;

Measure::Measure(int d)
	: d1(0), d2(0),
	valuesPtr(new MKL_Complex8[2 * d]),
	colPtr(new int[2 * d]),
	pBPtr(new int[2 * d]),
	pEPtr(new int[2 * d])
{
	for (int i = 0; i < 2 * d; i++)
		valuesPtr[i] = { 0,0 };
	for (int i = 0; i < 2 * d; i++)
		colPtr[i] = 0;
	for (int i = 0; i < 2 * d; i++)
		pBPtr[i] = 0;
	for (int i = 0; i < 2 * d; i++)
		pEPtr[i] = 0;
}
Measure::~Measure()
{
	delete[] valuesPtr;
	delete[] colPtr;
	delete[] pBPtr;
	delete[] pEPtr;
}
void Measure::measurePI_plus(int d2)
{
	for (int i = 0; i < d2; i++)
	{
		valuesPtr[i] = { 1,0 };
	}
	for (int i = d2; i < 2 * d2; i++)
	{
		valuesPtr[i] = { 0,0 };
	}

	for (int i = 0; i < 2 * d2; i++)
	{
		colPtr[i] = i;
	}

	for (int i = 0; i < 2 * d2; i++)
	{
		pBPtr[i] = i;
	}

	for (int i = 0; i < 2 * d2; i++)
	{
		pEPtr[i] = i + 1;
	}

}

void Measure::measurePI_minus(int d2)
{
	for (int i = 0; i < d2; i++)
	{
		valuesPtr[i] = { 0,0 };
	}
	for (int i = d2; i < 2 * d2; i++)
	{
		valuesPtr[i] = { 1,0 };
	}

	for (int i = 0; i < 2 * d2; i++)
	{
		colPtr[i] = i;
	}

	for (int i = 0; i < 2 * d2; i++)
	{
		pBPtr[i] = i;
	}

	for (int i = 0; i < 2 * d2; i++)
	{
		pEPtr[i] = i + 1;
	}

}

void Measure::measureIPI_plus(int d1, int d2)
{
	for (int i = 0; i < d1; i++)
	{
		for (int j = 0; j < d2; j++)
		{
			valuesPtr[i * 2 * d2 + j] = { 1, 0 };
		}

		for (int j = d2; j < 2 * d2; j++)
		{
			valuesPtr[i * 2 * d2 + j] = { 0, 0 };
		}
	}

	for (int i = 0; i < 2 * d1 * d2; i++)
	{
		colPtr[i] = i;
	}

	for (int i = 0; i < 2 * d1 * d2; i++)
	{
		pBPtr[i] = i;
	}

	for (int i = 0; i < 2 * d1 * d2; i++)
	{
		pEPtr[i] = i + 1;
	}
}

void Measure::measureIPI_minus(int d1, int d2)
{
	for (int i = 0; i < d1; i++)
	{
		for (int j = 0; j < d2; j++)
		{
			valuesPtr[i * 2 * d2 + j] = { 0, 0 };
		}

		for (int j = d2; j < 2 * d2; j++)
		{
			valuesPtr[i * 2 * d2 + j] = { 1, 0 };
		}
	}

	for (int i = 0; i < 2 * d1 * d2; i++)
	{
		colPtr[i] = i;
	}

	for (int i = 0; i < 2 * d1 * d2; i++)
	{
		pBPtr[i] = i;
	}

	for (int i = 0; i < 2 * d1 * d2; i++)
	{
		pEPtr[i] = i + 1;
	}

}

void Measure::measureIP_plus(int d1)
{
	for (int i = 0; i < 2 * d1; i++)
	{
		if (i % 2 == 0)
		{
			valuesPtr[i] = { 1,0 };
		}
		else
		{
			valuesPtr[i] = { 0,0 };
		}
	}


	for (int i = 0; i < 2 * d1; i++)
	{
		colPtr[i] = i;
	}

	for (int i = 0; i < 2 * d1; i++)
	{
		pBPtr[i] = i;
	}

	for (int i = 0; i < 2 * d1; i++)
	{
		pEPtr[i] = i + 1;
	}
}


void Measure::measureIP_minus(int d1)
{
	for (int i = 0; i < 2 * d1; i++)
	{
		if (i % 2 == 0)
		{
			valuesPtr[i] = { 0,0 };
		}
		else
		{
			valuesPtr[i] = { 1,0 };
		}
	}

	for (int i = 0; i < 2 * d1; i++)
	{
		colPtr[i] = i;
	}

	for (int i = 0; i < 2 * d1; i++)
	{
		pBPtr[i] = i;
	}

	for (int i = 0; i < 2 * d1; i++)
	{
		pEPtr[i] = i + 1;
	}
}


MKL_Complex8 *Measure::GetValues()
{
	return valuesPtr;
}
int *Measure::GetCol()
{
	return colPtr;
}
int *Measure::GetPB()
{
	return pBPtr;
}
int *Measure::GetPE()
{
	return pEPtr;
}
