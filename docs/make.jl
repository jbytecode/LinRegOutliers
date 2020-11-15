using Documenter

using Pkg
Pkg.activate("../")
using LinRegOutliers

push!(LOAD_PATH,"../src/")
makedocs(
	sitename="LinRegOutliers",
	pages=[
		"datasets.md",
		"types.md",
		"diagnostics.md",
		"algorithms.md"
	]
)


