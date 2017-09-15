// MIT License
//
// Copyright (c) 2017 David Gutzwiller, NUMECA Int.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files (the "Software"), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software,  and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all 
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
// SOFTWARE.

#include <iostream>
#include <cstdlib>
#include <cmath>

#pragma acc routine seq
int bisection(const float target, const float* values, const int nbValues)
{
  if (target <  values[0])          return 0;
  if (target >= values[nbValues-1]) return nbValues-1;
  int iMin = 0;
  int iMax = nbValues-1;
  int iMid = (iMin+iMax)/2;
  while ((iMax-iMin) > 1)
  {
    if (target > values[iMid])
      iMin = iMid;
    else
      iMax = iMid;
    iMid = (iMin+iMax)/2;
  }
  return iMin;
}

class LookupTable2D
{
  public:
    LookupTable2D()
    {
      _nbDataX = 100;
      _nbDataY = 100;
      _xVals = new float[_nbDataX];
      _yVals = new float[_nbDataY];
      _zVals = new float*[_nbDataX];
      for (int i=0; i<_nbDataX; i++) _zVals[i] = new float[_nbDataY];

      // initialize the table based on the ideal gas law (p=rho*R*T)
      // In a real code this would be experimental data read from file
      float R = 287.0;
      for (int i=0; i<_nbDataX; i++)
      {
        _xVals[i] = i*i/2.0;
        for (int j=0; j<_nbDataY; j++)
        {
          _yVals[j] = j/5.0;
          _zVals[i][j] = _xVals[i]*R*_yVals[j];
        }
      }
    }
    
    ~LookupTable2D()
    {
      for (int i=0; i<_nbDataX; i++) delete [] _zVals[i];
      delete[] _zVals;
      delete[] _xVals;
      delete[] _yVals;
    }

    #pragma acc routine seq
    float interpolate(const float xTarget, const float yTarget)
    {
      int iMin = bisection(xTarget,_xVals,_nbDataX);
      int jMin = bisection(yTarget,_yVals,_nbDataY);
      float valueSum  = 0.0;
      float weightSum = 0.0;
      for (int i=0; i<2; i++)
      {
        for (int j=0; j<2; j++)
        {
          float weight = 1.0 / sqrtf(powf(xTarget-_xVals[iMin+i],2.0) + powf(yTarget-_yVals[jMin+j],2.0));
          weightSum += weight;
          valueSum  += weight*_zVals[iMin+i][jMin+j];
        }
      }
      return valueSum / weightSum;
    }

    void createDevice()
    {
      #pragma acc enter data copyin(this)
      #pragma acc enter data copyin(_zVals[0:_nbDataX][0:_nbDataY])
      #pragma acc enter data copyin(_xVals[0:_nbDataX])
      #pragma acc enter data copyin(_yVals[0:_nbDataY])
    }

    void deleteDevice()
    {
      #pragma acc exit data delete(_zVals[0:_nbDataX][0:_nbDataY])
      #pragma acc exit data delete(_xVals[0:_nbDataX])
      #pragma acc exit data delete(_yVals[0:_nbDataY])
      #pragma acc exit data delete(this)
    }
    
  private:
    float** _zVals;
    float*  _xVals;
    float*  _yVals;
    int      _nbDataX;
    int      _nbDataY;
};

int main(int argc,char *argv[])
{
  if (argc != 3)
  {
    std::cout << " ERROR, 2 arguments expected: int nbdata, int nbIter " << argc << std::endl;
    exit(1);
  }
  int nbData = std::atoi(argv[1]);
  int nbIter = std::atoi(argv[2]);

  // initialize random temperature and density values on the host
  LookupTable2D* presFromRhoTemp = new LookupTable2D();
  float* rho  = new float[nbData];
  float* temp = new float[nbData];
  float* pres = new float[nbData];
  srand(1);
  for (int i=0; i<nbData; i++)
  {
    temp[i] = rand()%400 + ((float) rand()) / (float) RAND_MAX;
    rho[i]  = rand()%4   + ((float) rand()) / (float) RAND_MAX;
    pres[i] = 0.0;
  }

  for (int iter=0; iter<nbIter; iter++)
  {
    presFromRhoTemp->createDevice();
    #pragma acc parallel loop \
     copy(pres[0:nbData],temp[0:nbData],rho[0:nbData]) \
     present(presFromRhoTemp)
    for (int i=0; i<nbData; i++)
    {
      pres[i] = presFromRhoTemp->interpolate(temp[i],rho[i]);
    }
    presFromRhoTemp->deleteDevice();
  }
  
  delete[] temp;
  delete[] rho;
  delete[] pres;
  delete   presFromRhoTemp;
}
