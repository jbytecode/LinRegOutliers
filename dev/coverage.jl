import Pkg

Pkg.activate("../")
Pkg.resolve()

import Coverage

Pkg.test(coverage = true)

tested, total = Coverage.process_folder("../") |> Coverage.get_summary

ratio = tested / total

Coverage.clean_folder("../")


@info "Coverage: $ratio"

