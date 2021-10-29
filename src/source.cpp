#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <omp.h>
#include <mkl_dfti.h>

#define w(N,k) (std::complex<el_type>(cos(2*Pi*k/N), -sin(2*Pi*k/N)))
#define dft_w(N,n,k) (std::complex<el_type>(cos(2*Pi*k*n/N), -sin(2*Pi*k*n/N)))

typedef double el_type;
typedef std::vector<std::complex<el_type>> t_complex_vector;
constexpr double Pi = 3.14159265359;

template <typename T>
T m_reverse(T a, int bit_len)
{
	T res = 0;
	for (int i = 0; i < bit_len; ++i)
	{
		bool bit = (a >> i) % 2;
		res |= bit << (bit_len - i - 1);
	}
	return res;
}

int my_log_2(int a)
{
	int res = 0;
	int tmp = 1;
	while (tmp != a)
	{
		tmp *= 2;
		res++;
	}
	return res;
}

template <typename T>
void pre_permutation_algorithm(T &src)//src.size() must be power of 2
{
	int n = src.size();
	int bit_length = my_log_2(n);
	T res(n);
	for (int i = 0; i < n; ++i)
	{
		res[m_reverse(i, bit_length)] = src[i];
	}
	src = res;
}

void m_FFT(t_complex_vector &src, t_complex_vector &res)
{
	int size = src.size();
	int iterations = my_log_2(size);
	int subsequence_size = 1;
	t_complex_vector pre_calculated_wnk(size / 2);
	t_complex_vector buffer(size);
	for (size_t i = 0; i < size / 2; ++i)
	{
		pre_calculated_wnk[i] = w(size, i);
	}
	if (&res != &src)
		res = src;
	pre_permutation_algorithm(res);
	for (int i = 0; i < iterations; ++i)
	{
		#pragma omp parallel for 
		for (int j = 0; j < size / (subsequence_size * 2); ++j)
		{
			for (int t = 0; t < subsequence_size; ++t)
			{
				buffer[j * subsequence_size * 2 + t] = res[j * subsequence_size * 2 + t] + pre_calculated_wnk[t * (size / (subsequence_size * 2))] * res[j * subsequence_size * 2 + subsequence_size + t];
				buffer[j * subsequence_size * 2 + subsequence_size + t] = res[j * subsequence_size * 2 + t] - pre_calculated_wnk[t * (size / (subsequence_size * 2))] * res[j * subsequence_size * 2 + subsequence_size + t];
			}
		}
		#pragma omp parallel for
		for (int k = 0; k < size; ++k)
		{
			res[k] = buffer[k];
		}
		subsequence_size *= 2;
	}
}

void DFT(t_complex_vector &x)
{
	int N = x.size();
	t_complex_vector res(N);
	for (int k = 0; k < N; ++k)
	{
		for (int n = 0; n < N; ++n)
		{
			res[k] += dft_w(N, n, k) * x[n];
		}
	}
	x = res;
}

bool check(el_type eps, t_complex_vector &a, t_complex_vector &b)
{
	for (size_t i = 0; i < a.size(); ++i)
	{
		if(abs(a[i].imag() - b[i].imag()) > eps || abs(a[i].real() - b[i].real()) > eps)
		{
			return false;
		}
	}
	return true;
}

void mkl_fft(t_complex_vector& in, t_complex_vector &out) 
{
	DFTI_DESCRIPTOR_HANDLE descriptor;
	MKL_LONG status;
	DFTI_CONFIG_VALUE presicion;
	DFTI_CONFIG_VALUE placement;
	if (typeid(el_type) == typeid(double))
	{
		presicion = DFTI_CONFIG_VALUE::DFTI_DOUBLE;
	}
	else if (typeid(el_type) == typeid(float))
	{
		presicion = DFTI_CONFIG_VALUE::DFTI_SINGLE;
	}
	if (&in == &out)
	{
		placement = DFTI_CONFIG_VALUE::DFTI_INPLACE;
	}
	else if (&in != &out)
	{
		placement = DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE;
	}
	status = DftiCreateDescriptor(&descriptor, presicion, DFTI_COMPLEX, 1, in.size()); //Specify size and precision
	status = DftiSetValue(descriptor, DFTI_PLACEMENT, placement); //In/out of place FFT
	status = DftiCommitDescriptor(descriptor); //Finalize the descriptor
	status = DftiComputeForward(descriptor, in.data(), out.data()); //Compute the Forward FFT
	status = DftiFreeDescriptor(&descriptor); //Free the descriptor
}

int main()
{
	int n = 8192 * 128;
	double mkl_time;
	double my_fft_time;
	std::ofstream file_out("../out.xls");
	std::vector<std::complex<el_type>> x(n, std::complex<el_type>());
	std::vector<std::complex<el_type>> y(n, std::complex<el_type>());
	file_out << "size" << '\t' << "MKL_FFT" << '\t' << "MY_FFT" << std::endl;
	for (int i = 0; i < n; ++i)
	{
		y[i] = x[i] = std::complex<el_type>(rand() % 20 / 10. - 1, rand() % 20 / 10. - 1);
	}
	for (int i = 0; i < 5; ++i)
	{
		mkl_time = omp_get_wtime();
		mkl_fft(y, y);
		mkl_time = omp_get_wtime() - mkl_time;
		my_fft_time = omp_get_wtime();
		m_FFT(x, x);
		my_fft_time = omp_get_wtime() - my_fft_time;
		file_out << n << '\t' << mkl_time << '\t' << my_fft_time << std::endl;
		n *= 2;
		x.resize(n);
		y.resize(n);
		for (int i = 0; i < n; ++i)
		{
			y[i] = x[i] = std::complex<el_type>(rand() % 20 / 10. - 1, rand() % 20 / 10. - 1);
		}
	}
	return 0;
}