#include <cmath>

#include <armadillo>

using arma::uword;
using arma::uvec;
using arma::field;

class mcmc_run {
	public:
		mcmc_run(n_vars): n_vars_(n_vars), state_(arma::vec(n_vars, arma::fill::zeros)), t_(0) {}

		inline float loglikelihood(arma::vec observations){
			return -1.0; // tbd
		}
		void step(){} 
		// Monte Carlo step. changes state. 
		// generate rand var, filter with likelihood. 

	private:
		uword t_;
		uword n_vars_;
		arma::vec state_;
};


class T_grid {
	// 2D grid for T values
	public:
		T_grid(N, T_scale) : 
			N_(N), 
			T_scale_(T_scale),
			run_ptrs_(N,N), // default initializes to nullptr's
			running_(N,N)   // default initializes to false's
			{}

		inline uvec symmetrize_pt(arma::uvec ij){
			if(ij[0] < ij[1]) {
				return ij;
			} else {
				return arma::uvec({ij[1], ij[0]})
			}
		}

		inline uvec get_gridpoint(T1, T2){
			return symmetrize_pt(
					uvec({
						std::round((T1 / T_scale) * N_),
						std::round((T2 / T_scale) * N_)
					})
				);
		}

		bool is_running(uword i, uword j){ return running_[i,j]; }
		void set_running(uword i, uword j){ running_[i,j] = true; }

		mcmc_run* get_run_ptr(uword i, uword j){
			// tbd should probably check for nullptr first and handle that error
			return run_ptrs_[i,j];
		}
		void set_run_ptr(uword i, uword j, mcmc_run* run_ptr){ run_ptrs_[i,j] = run_ptr; }

	private:
		uword N_;
		field<mcmc_run*> run_ptrs_;
		field<bool> running_;
		float T_scale_;
};

class event {
	public:
		event(uword i, uword j) : i_(i), j_(j) {}
		float T1(){ return 0.0; }
		float T2(){ return 0.0; }
		mcmc_run* run_chain(){
			// ...
		}
	private:
};

class scheduler {
	public:
		scheduler(N, T_scale) : t_grid_(N, T_scale) {}
	private:
		*T_grid t_grid_;
};