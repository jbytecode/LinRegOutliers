using Documenter, LinRegOutliers


makedocs(
         format = Documenter.HTML(
                                  prettyurls = get(ENV, "CI", nothing) == "true",
                                  collapselevel = 2,
                                  # assets = ["assets/favicon.ico", "assets/extra_styles.css"],
                                 ),
         sitename="LinRegOutliers",
         authors = "Mehmet Hakan Satman <mhsatman@gmail.com>, Shreesh Adiga <16567adigashreesh@gmail.com>, Guillermo Angeris <angeris@stanford.edu>, Emre Akadal <emre.akadal@istanbul.edu.tr>",
         pages = [
                  "Algorithms" => "algorithms.md",
                  "Diagnostics" =>  "diagnostics.md",
                  "Types" => "types.md",
                  "Datasets" => "datasets.md"
                 ]
        )


deploydocs(
   repo = "github.com/jbytecode/LinRegOutliers",
)



