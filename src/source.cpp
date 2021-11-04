#include <iostream>
#include <iomanip>
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

void m_FFT(t_complex_vector &src, t_complex_vector &res, bool mthread_param)
{
	size_t size = src.size();
	int iterations = my_log_2(size);
	size_t subsequence_size = 1;
	t_complex_vector buffer(size);
	if (&res != &src)
		res = src;
	pre_permutation_algorithm(res);
	for (int i = 0; i < iterations; ++i)
	{
#pragma omp parallel for if (mthread_param == 1 && size >= 1024)
		for (int j = 0; j < size / (subsequence_size * 2); ++j)
		{
			for (int t = 0; t < subsequence_size; ++t)
			{
				buffer[j * subsequence_size * 2 + t] = res[j * subsequence_size * 2 + t] + w(size, t * (size / (subsequence_size * 2))) * res[j * subsequence_size * 2 + subsequence_size + t];
				buffer[j * subsequence_size * 2 + subsequence_size + t] = res[j * subsequence_size * 2 + t] - w(size, t * (size / (subsequence_size * 2)))* res[j * subsequence_size * 2 + subsequence_size + t];
			}
		}
		for (int k = 0; k < size; ++k)
		{
			res[k] = buffer[k];
		}
		subsequence_size *= 2;
	}
}

void m_FFT_vectorized(std::vector<el_type> &src_real, std::vector<el_type> &src_im, std::vector<el_type> &res_real, std::vector<el_type> &res_im, bool mthread_param)
{
	size_t size = src_real.size();
	int iterations = my_log_2(size);
	size_t subsequence_size = 1;
	std::vector<el_type> buffer_real(size);
	std::vector<el_type> buffer_imag(size);
	el_type tmp_real;
	el_type tmp_im;
	if (&src_real != &res_real)
		res_real = src_real;
	if (&src_im != &res_im)
		res_im = src_im;
	pre_permutation_algorithm(res_real);
	pre_permutation_algorithm(res_im);
	for (int i = 0; i < iterations; ++i)
	{
#pragma omp parallel for if (mthread_param == 1 && size >= 1024)
		for (int j = 0; j < size / (subsequence_size * 2); ++j)
		{
			for (int t = 0; t < subsequence_size; ++t)
			{
				tmp_real = cos(2 * (Pi / subsequence_size / 2)*t);
				tmp_im = -sin(2 * (Pi / subsequence_size / 2)*t);
				buffer_real[j * subsequence_size * 2 + t] = res_real[j * subsequence_size * 2 + t] + (tmp_real * res_real[j * subsequence_size * 2 + subsequence_size + t] - tmp_im * res_im[j * subsequence_size * 2 + subsequence_size + t]);
				buffer_imag[j * subsequence_size * 2 + t] = res_im[j * subsequence_size * 2 + t] + (tmp_im * res_real[j * subsequence_size * 2 + subsequence_size + t] + tmp_real * res_im[j * subsequence_size * 2 + subsequence_size + t]);

				buffer_real[j * subsequence_size * 2 + subsequence_size + t] = res_real[j * subsequence_size * 2 + t] - (tmp_real * res_real[j * subsequence_size * 2 + subsequence_size + t] - tmp_im * res_im[j * subsequence_size * 2 + subsequence_size + t]);
				buffer_imag[j * subsequence_size * 2 + subsequence_size + t] = res_im[j * subsequence_size * 2 + t] - (tmp_im * res_real[j * subsequence_size * 2 + subsequence_size + t] + tmp_real * res_im[j * subsequence_size * 2 + subsequence_size + t]);
			}
		}
		for (int k = 0; k < size; ++k)
		{
			res_real[k] = buffer_real[k];
			res_im[k] = buffer_imag[k];
		}
		subsequence_size *= 2;
	}
}

void m_two_dimensional_FT(t_complex_vector &src, size_t n, size_t m, bool mthread_param)
{
	t_complex_vector temp(m);
	#pragma omp parallel for if (mthread_param == 1)
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			temp[j] = src[i*m + j];
		}
		m_FFT(temp, temp, 1);
		for (int j = 0; j < m; ++j)
		{
			src[i*m + j] = temp[j];
		}
	}
	temp.resize(n);
	#pragma omp parallel for if (mthread_param == 1)
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			temp[j] = src[i + j * m];
		}
		m_FFT(temp, temp, 1);
		for (int j = 0; j < n; ++j)
		{
			src[i + j * m] = temp[j];
		}
	}
}

void m_two_dimensional_DFT(t_complex_vector &src, size_t N, size_t M)
{
	t_complex_vector res(N*M);
	for (size_t k = 0; k < N; ++k)
	{
		for (size_t l = 0; l < M; ++l)
		{
			for (size_t m = 0; m < M; ++m)
			{
				for (size_t n = 0; n < N; ++n)
				{
					res[k*M + l] += src[n*M + m] * dft_w(M, m, l) * dft_w(N, n, k);
				}
			}
		}
	}
	src = res;
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
		if (abs(a[i].imag() - b[i].imag()) > eps || abs(a[i].real() - b[i].real()) > eps)
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
	DFTI_CONFIG_VALUE precision;
	DFTI_CONFIG_VALUE placement;
	if (typeid(el_type) == typeid(double))
	{
		precision = DFTI_CONFIG_VALUE::DFTI_DOUBLE;
	}
	else if (typeid(el_type) == typeid(float))
	{
		precision = DFTI_CONFIG_VALUE::DFTI_SINGLE;
	}
	if (&in == &out)
	{
		placement = DFTI_CONFIG_VALUE::DFTI_INPLACE;
	}
	else if (&in != &out)
	{
		placement = DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE;
	}
	status = DftiCreateDescriptor(&descriptor, precision, DFTI_COMPLEX, 1, in.size()); //Specify size and precision
	status = DftiSetValue(descriptor, DFTI_PLACEMENT, placement); //In/out of place FFT
	status = DftiCommitDescriptor(descriptor); //Finalize the descriptor
	status = DftiComputeForward(descriptor, in.data(), out.data()); //Compute the Forward FFT
	status = DftiFreeDescriptor(&descriptor); //Free the descriptor
}

void test_1()
{
	size_t n = 32768;
	double mkl_time;
	double my_fft_time;
	double my_vectorized_fft_time;
	std::ofstream file_out("../out.xls");
	file_out << std::fixed << std::showpoint << std::setprecision(3);
	t_complex_vector x(n);
	t_complex_vector y(n);
	std::vector<el_type> z_real(n);
	std::vector<el_type> z_im(n);
	file_out << "size" << '\t' << "MKL_FFT" << '\t' << "MY_FFT" << '\t' << "MY_FFT_VECTORIZED" << std::endl;
	for (int i = 0; i < n; ++i)
	{
		el_type real = rand() % 20 / 10. - 1;
		el_type imag = rand() % 20 / 10. - 1;
		y[i] = x[i] = std::complex<el_type>(real, imag);
		z_real[i] = real;
		z_im[i] = imag;
	}
	for (int i = 0; i < 12; ++i)
	{
		mkl_time = omp_get_wtime();
		mkl_fft(y, y);
		mkl_time = omp_get_wtime() - mkl_time;
		my_fft_time = omp_get_wtime();
		m_FFT(x, x, 1);
		my_fft_time = omp_get_wtime() - my_fft_time;
		my_vectorized_fft_time = omp_get_wtime();
		m_FFT_vectorized(z_real, z_im, z_real, z_im, 1);
		my_vectorized_fft_time = omp_get_wtime() - my_vectorized_fft_time;
		file_out << n << '\t' << mkl_time << '\t' << my_fft_time << '\t' << my_vectorized_fft_time << std::endl;
		n *= 2;
		x.resize(n);
		y.resize(n);
		z_real.resize(n);
		z_im.resize(n);
		for (int i = 0; i < n; ++i)
		{
			el_type real = rand() % 20 / 10. - 1;
			el_type imag = rand() % 20 / 10. - 1;
			y[i] = x[i] = std::complex<el_type>(real, imag);
			z_real[i] = real;
			z_im[i] = imag;
		}
	}
}
void test2()
{
	size_t n = 64;
	size_t m = 64;
	t_complex_vector x(n*m);
	t_complex_vector y(n*m);
	for (int i = 0; i < n * m; ++i)
	{
		el_type real = rand() % 20 / 10. - 1;
		el_type imag = rand() % 20 / 10. - 1;
		y[i] = x[i] = std::complex<el_type>(real, imag);
	}
	m_two_dimensional_DFT(y, n, m);
	m_two_dimensional_FT(x, n, m, 0);
	std::cout << check(1.e-7, x, y) << std::endl;
}
void test3()
{
	size_t n = 32;
	size_t m = 32;
	t_complex_vector x(n*m);
	t_complex_vector y(n*m);
	double dft_time;
	double fft_time;
	std::ofstream file_out("../out.xls");
	file_out << std::fixed << std::showpoint << std::setprecision(3);
	file_out << "size" << '\t' << "2D_DFT" << '\t' << "2D_FT_USING_1DFFT" << std::endl;
	for (int i = 0; i < n * m; ++i)
	{
		el_type real = rand() % 20 / 10. - 1;
		el_type imag = rand() % 20 / 10. - 1;
		y[i] = x[i] = std::complex<el_type>(real, imag);
	}
	for (int i = 0; i < 4; ++i)
	{
		dft_time = omp_get_wtime();
		m_two_dimensional_DFT(y, n, m);
		dft_time = omp_get_wtime() - dft_time;
		fft_time = omp_get_wtime();
		m_two_dimensional_FT(x, n, m, 0);
		fft_time = omp_get_wtime() - fft_time;
		file_out << n*m << '\t' << dft_time << '\t' << fft_time << std::endl;
		n *= 2;
		m *= 2;
		x.resize(n*m);
		y.resize(n*m);
		for (int i = 0; i < n * m; ++i)
		{
			el_type real = rand() % 20 / 10. - 1;
			el_type imag = rand() % 20 / 10. - 1;
			y[i] = x[i] = std::complex<el_type>(real, imag);
		}
		std::cout << check(1.e-7, x, y);
	}
}

int main()
{
	test3();
	while (1);
	return 0;
}