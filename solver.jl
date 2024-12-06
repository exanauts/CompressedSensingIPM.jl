using MadNLP

function ipm_solve!(solver::MadNLP.MadNLPSolver)
    MadNLP.print_init(solver)
    MadNLP.initialize!(solver)
    MadNLP.regular!(solver)
    return MadNLP.MadNLPExecutionStats(solver)
end
