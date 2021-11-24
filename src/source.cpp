#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <omp.h>
#include <mkl_dfti.h>
#include <bitset>

#define w(N,k) (std::complex<el_type>(cos(2*Pi*k/N), -sin(2*Pi*k/N)))
#define dft_w(N,n,k) (std::complex<el_type>(cos(2*Pi*k*n/N), -sin(2*Pi*k*n/N)))

typedef double el_type;
typedef std::vector<std::complex<el_type>> t_complex_vector;
constexpr double Pi = 3.14159265359;

template <typename T>
inline T m_reverse(T a, int bit_len)
{
	T res = 0;
	for (int i = 0; i < bit_len; ++i)
	{
		bool bit = (a >> i) % 2;
		res |= bit << (bit_len - i - 1);
	}
	return res;
}

inline int my_log_2(int a)
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
	for (int i = 1; i < n - 1; ++i)
	{
		size_t j = m_reverse(i, bit_length);
		if (j <= i) continue;
		std::swap(src[i], src[j]);
	}
}

void m_FFT(t_complex_vector &src, t_complex_vector &res, bool mthread_param)
{
	size_t size = src.size();
	int iterations = my_log_2(size);
	size_t subsequence_size = 1;
	std::complex<el_type> temp_first;
	std::complex<el_type> temp_second;
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
				temp_first = res[j * subsequence_size * 2 + t] + w(size, t * (size / (subsequence_size * 2))) * res[j * subsequence_size * 2 + subsequence_size + t];
				temp_second = res[j * subsequence_size * 2 + t] - w(size, t * (size / (subsequence_size * 2)))* res[j * subsequence_size * 2 + subsequence_size + t];
				res[j * subsequence_size * 2 + t] = temp_first;
				res[j * subsequence_size * 2 + subsequence_size + t] = temp_second;
			}
		}
		subsequence_size *= 2;
	}
}

void m_FFT_vectorized_cached(std::vector<el_type> &src_real, std::vector<el_type> &src_im, std::vector<el_type> &res_real, std::vector<el_type> &res_im, size_t max_group_size, bool mthread_param)
{
	const size_t size = src_real.size();
	size_t global_subsequence_size = 1;
	const size_t bit_length = my_log_2(size);
	const int iterations = my_log_2(size);
	const size_t group_size = size < max_group_size ? size : max_group_size;
	const size_t sincos_buffer_size = group_size / 2;
	const size_t passes = my_log_2(group_size);
	const size_t epochs = iterations / passes;
	const el_type constant_factor = 2 * Pi / size;
	size_t adress_prefix_bit_mask;
	size_t adress_postfix_bit_mask;
	size_t exp_shift = iterations - 1;
	std::vector<el_type> buf_real(group_size * omp_get_max_threads());
	std::vector<el_type> buf_imag(group_size * omp_get_max_threads());
	std::vector<el_type> pre_calc_sin(2*sincos_buffer_size - 1);
	std::vector<el_type> pre_calc_cos(2*sincos_buffer_size - 1);
	if (&src_real != &res_real)
		res_real = src_real;
	if (&src_im != &res_im)
		res_im = src_im;
	#pragma omp parallel for if(mthread_param == 1)
	for (size_t i = 1; i < size-1; ++i)
	{
		size_t j = m_reverse(i, bit_length);
		if (j <= i) continue;
		std::swap(res_real[i], res_real[j]);
		std::swap(res_im[i], res_im[j]);
	}
	for (size_t i = 0; i < passes; ++i)
	{
		#pragma omp simd
		for (size_t j = 0; j < (1 << i); ++j)
		{
			el_type argument = Pi*j/(1 << i);
			pre_calc_cos[(1 << i) - 1 + j] = cos(argument);
			pre_calc_sin[(1 << i) - 1 + j] = -sin(argument);
		}
	}
	for (size_t epoch_counter = 0; epoch_counter < epochs; ++epoch_counter)
	{
		adress_prefix_bit_mask = (1 << epoch_counter * passes) - 1;
		adress_postfix_bit_mask = ~adress_prefix_bit_mask;
		#pragma omp parallel for if(mthread_param == 1)
		for (int group_counter = 0; group_counter < size / group_size; ++group_counter)
		{
			size_t subsequence_size = 1;
			size_t buf_start_adress = group_size * omp_get_thread_num();
			size_t first_butterfly_first_input_adress = adress_prefix_bit_mask & group_counter | ((group_counter & adress_postfix_bit_mask) << passes);
			size_t exp_postfix;
			size_t cur_group;
			for (int i = 0; i < group_size; ++i)//load to cache
			{
				buf_real[buf_start_adress + i] = res_real[first_butterfly_first_input_adress + global_subsequence_size * i];
				buf_imag[buf_start_adress + i] = res_im[first_butterfly_first_input_adress + global_subsequence_size * i];
			}
			for (size_t pass_counter = 0; pass_counter < passes; ++pass_counter)
			{
				size_t butterfly_adress_mask = (1 << pass_counter) - 1;
				exp_postfix = (group_counter & adress_prefix_bit_mask) << (exp_shift - pass_counter);
				for (int j = 0; j < group_size / (subsequence_size * 2); ++j)
				{
					#pragma omp simd
					for (int t = 0; t < subsequence_size; ++t)
					{
						el_type temp_cos;
						el_type temp_sin;
						size_t t_adress = buf_start_adress + j * subsequence_size * 2 + t;
						if (epoch_counter == 0)
						{
							size_t sincos_index = (1 << pass_counter) - 1;
							temp_cos = pre_calc_cos[sincos_index + t];
							temp_sin = pre_calc_sin[sincos_index + t];
						}
						else
						{
							el_type argument = constant_factor * (exp_postfix | ((t & butterfly_adress_mask) << epoch_counter * passes + exp_shift - pass_counter));
							temp_cos = cos(argument);
							temp_sin = -sin(argument);
						}
						el_type temp_first = (temp_cos * buf_real[t_adress + subsequence_size] - temp_sin * buf_imag[t_adress + subsequence_size]);
						el_type temp_second = (temp_sin * buf_real[t_adress + subsequence_size] + temp_cos * buf_imag[t_adress + subsequence_size]);
						el_type temp_real_t = buf_real[t_adress] + temp_first;
						el_type temp_imag_t = buf_imag[t_adress] + temp_second;
						el_type temp_real_ss_plus_t = buf_real[t_adress] - temp_first;
						el_type temp_imag_ss_plus_t = buf_imag[t_adress] - temp_second;
						buf_real[t_adress] = temp_real_t;
						buf_imag[t_adress] = temp_imag_t;
						buf_real[t_adress + subsequence_size] = temp_real_ss_plus_t;
						buf_imag[t_adress + subsequence_size] = temp_imag_ss_plus_t;
					}
				}
				subsequence_size = subsequence_size << 1;
			}
			for (int i = 0; i < group_size; ++i)//load to main memory
			{
				res_real[first_butterfly_first_input_adress + global_subsequence_size * i] = buf_real[buf_start_adress + i];
				res_im[first_butterfly_first_input_adress + global_subsequence_size * i] = buf_imag[buf_start_adress + i];
			}
		}
		global_subsequence_size = global_subsequence_size << passes;
		exp_shift -= passes;
	}
	if (iterations % passes != 0)
	{
		size_t subsequence_size = global_subsequence_size;
		for (size_t i = iterations - epochs * passes + 1; i < iterations; ++i)
		{
			#pragma omp parallel for if(mthread_param == 1 && size >= 1024)
			for (int j = 0; j < size / (subsequence_size * 2); ++j)
			{
				#pragma omp simd
				for (int t = 0; t < subsequence_size; ++t)
				{
					size_t t_adress = j * subsequence_size * 2 + t;
					el_type temp_cos = cos(Pi / subsequence_size*t);
					el_type temp_sin = -sin(Pi / subsequence_size*t);
					el_type temp_first = temp_cos * res_real[t_adress + subsequence_size] - temp_sin * res_im[t_adress + subsequence_size];
					el_type temp_second = temp_sin * res_real[t_adress + subsequence_size] + temp_cos * res_im[t_adress + subsequence_size];
					el_type temp_real_t = res_real[t_adress] + temp_first;
					el_type temp_imag_t = res_im[t_adress] + temp_second;
					el_type temp_real_ss_plus_t = res_real[t_adress] - temp_first;
					el_type temp_imag_ss_plus_t = res_im[t_adress] - temp_second;
					res_real[t_adress] = temp_real_t;
					res_im[t_adress] = temp_imag_t;
					res_real[t_adress + subsequence_size] = temp_real_ss_plus_t;
					res_im[t_adress + subsequence_size] = temp_imag_ss_plus_t;
				}
			}
			subsequence_size *= 2;
		}
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
	src.swap(res);
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
void test_2()
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
void test_3()
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
void test_4()
{
	std::ofstream file_out("../out.xls");
	file_out << std::fixed << std::showpoint << std::setprecision(3);
	size_t n = 16'777'216;
	size_t cache_size = 32;
	double cached_time;
	double mkl_time;
	std::vector<el_type> z_real(n);
	std::vector<el_type> z_im(n);
	std::vector<el_type> q_real(n);
	std::vector<el_type> q_im(n);
	t_complex_vector x(n);
	t_complex_vector y(n);
	for (int i = 0; i < n; ++i)
	{
		el_type real = rand() % 20 / 10. - 1;
		el_type imag = rand() % 20 / 10. - 1;
		x[i] = std::complex<el_type>(real, imag);
	}
	mkl_time = omp_get_wtime();
	mkl_fft(x, x);
	mkl_time = omp_get_wtime() - mkl_time;
	file_out << mkl_time << std::endl;
	for (int i = 0; i < 15; ++i)
	{
		for (int i = 0; i < n; ++i)
		{
			el_type real = rand() % 20 / 10. - 1;
			el_type imag = rand() % 20 / 10. - 1;
			z_real[i] = real;
			z_im[i] = imag;
			x[i] = std::complex<el_type>(real, imag);
		}
		cached_time = omp_get_wtime();
		m_FFT_vectorized_cached(z_real, z_im, z_real, z_im, cache_size, 1);
		cached_time = omp_get_wtime() - cached_time;
		m_FFT(x, x, 1);
		for (int i = 0; i < n; ++i)
		{
			y[i] = std::complex<el_type>(z_real[i], z_im[i]);
		}
		file_out << cache_size << '\t' << '\t' << cached_time << std::endl;
		std::cout << check(1.e-7, x, y) << '\t' << i << std::endl;
		cache_size *= 2;
	}
}
void test_5()
{
	size_t n = 16'777'216;
	size_t cache_size = 32;
	n = 8;
	cache_size = 4;
	std::vector<el_type> z_real(n);
	std::vector<el_type> z_im(n);
	t_complex_vector x(n);
	t_complex_vector y(n);
	for (int i = 0; i < n; ++i)
	{
		el_type real = rand() % 20 / 10. - 1;
		el_type imag = rand() % 20 / 10. - 1;
		z_real[i] = i;
		z_im[i] = 0;
		x[i] = std::complex<el_type>(i, 0);
	}
	m_FFT_vectorized_cached(z_real, z_im, z_real, z_im, cache_size, 0);
	m_FFT(x, x, 0);
	for (int i = 0; i < n; ++i)
	{
		y[i] = std::complex<el_type>(z_real[i], z_im[i]);
	}
	std::cout << check(1.e-7, x, y) << std::endl;
}

int main()
{
	test_4();
	while (1);
	return 0;
}