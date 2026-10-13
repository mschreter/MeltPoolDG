#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  template <typename number,
            typename VectorType = dealii::LinearAlgebra::distributed::Vector<number>>
  class NewtonRaphsonSolver
  {
  public:
    std::function<void(const VectorType &evaluation_point, VectorType &residual)> residual = {};
    std::function<int(const VectorType &rhs, VectorType &dst)> solve_with_jacobian         = {};
    std::function<void(VectorType &v)>                         reinit_vector               = {};
    std::function<void(VectorType &v)>                         distribute_constraints      = {};

    // TODO: remove and replace by standard l2_norm() (?)
    std::function<number()> norm_of_solution_vector = {};

  private:
    const NonlinearSolverData<number> nlsolve_data;

    dealii::ConditionalOStream pcout;

    const int max_number_of_iterations;
    number    residual_tolerance;
    number    field_correction_tolerance;

    VectorType rhs;
    VectorType solution_update;

    int                iteration_counter     = 0;
    int                linear_iter_acc       = 0;
    number             initial_residual_norm = 0.0;
    std::ostringstream str_;

  public:
    explicit NewtonRaphsonSolver(const NonlinearSolverData<number> &nlsolve_data);

    void
    solve(VectorType &solution);

    const VectorType &
    get_residual() const
    {
      return rhs;
    }

    const VectorType &
    get_solution_update() const
    {
      return solution_update;
    }

  private:
    void
    solve_increment(const VectorType &current_solution);

    number
    suggest_new_time_increment();

    void
    set_tolerances_to_alternative_values();

    bool
    is_converged(const VectorType &solution);

    void
    print_header() const;

    std::string
    print_checkmark(bool is_converged) const;
  };

} // namespace MeltPoolDG
