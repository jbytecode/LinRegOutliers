@testset "Diagnostics" begin

    @testset "dffit - single case" begin
        # Since this regression setting is deterministic,
        # omission of a single observation has not an effect on
        # the predicted response.
        myeps = 10.0^(-6.0)
        dataset = DataFrame(x = [1.0, 2, 3, 4, 5], y = [2.0, 4, 6, 8, 10])
        setting = createRegressionSetting(@formula(y ~ x), dataset)
        n, _ = size(dataset)
        for i = 1:n
            mydiff = dffit(setting, i)
            @test abs(mydiff) < myeps
        end
    end

    @testset "dffit - for all observations" begin
        # Since this regression setting is deterministic,
        # omission of a single observation has not an effect on
        # the predicted response.
        myeps = 10.0^(-6.0)
        dataset = DataFrame(x = [1.0, 2, 3, 4, 5], y = [2.0, 4, 6, 8, 10])
        setting = createRegressionSetting(@formula(y ~ x), dataset)
        stats = dffit(setting)
        for element in stats
            @test abs(element) < myeps
        end
    end

    @testset "hat matrix" begin
        X = [1.0 1.0; 1.0 2.0; 1.0 3.0; 1.0 4.0; 1.0 5.0]
        hats_real = X * inv(X' * X) * X'
        dataset = DataFrame(x = [1.0, 2, 3, 4, 5], y = [2.0, 4, 6, 8, 10])
        setting = createRegressionSetting(@formula(y ~ x), dataset)
        hats_calculated = hatmatrix(setting)

        @test hats_real == hats_calculated
    end

    @testset "studentized residuals" begin
        dataset = DataFrame(
            x = Float64[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            y = Float64[2.1, 3.9, 6.2, 7.8, 11.0, 18.0],
        )
        setting = createRegressionSetting(@formula(y ~ x), dataset)
        resi = studentizedResiduals(setting)
        @test abs(resi[1]) < 1.5
        @test abs(resi[2]) < 1.5
        @test abs(resi[3]) < 1.5
        @test abs(resi[4]) < 1.5
        @test abs(resi[5]) < 1.5
        @test abs(resi[6]) > 1.5
    end


    @testset "adjusted residuals" begin
        eps = 0.0001
        dataset = DataFrame(x = [1.0, 2, 3, 4, 5], y = [2.0, 4, 6, 8, 10])
        setting = createRegressionSetting(@formula(y ~ x), dataset)
        resi = adjustedResiduals(setting)
        for element in resi
            @test abs(element) < eps
        end
    end

    @testset "Cook's distance - Phone data" begin
        eps = 0.00001
        setting = createRegressionSetting(@formula(calls ~ year), phones)
        knowncooks = [
            0.005344774190779771,
            0.0017088194691033181,
            0.00016624914057961155,
            3.16444525831206e-5,
            0.0005395058666404081,
            0.0014375008774859539,
            0.0024828140956511258,
            0.0036279720445167277,
            0.004357605989540906,
            0.005288503758364767,
            0.006313578057565415,
            0.0076561205696857254,
            0.009568574875389256,
            0.009970039008782357,
            0.02610396373381051,
            0.029272523880917646,
            0.05091236198400663,
            0.08176555044049343,
            0.14380266904640235,
            0.26721539425047447,
            0.051205153558783356,
            0.13401084683481085,
            0.16860324592350226,
            0.2172819114905912,
        ]
        cookdists = cooks(setting)
        @test map((x, y) -> abs(x - y) < eps, cookdists, knowncooks) == trues(24)
    end

    @testset "Cook's distance - Cutoff" begin 
        eps = 0.001
        sett = createRegressionSetting(@formula(calls ~ year), phones)
        result = cooksoutliers(sett, alpha = 0.5)
        @test result isa Dict 
        @test isapprox(result["cutoff"], 0.715452, atol = eps) 
        potentials = result["potentials"]
        @test potentials isa Vector 
        @test length(potentials) == 0
    end 


    @testset "Jacknifed standard error of regression" begin
        eps = 0.00001
        dataset = DataFrame(x = [1.0, 2, 3, 4, 5000], y = [2.0, 4, 6, 8, 1000])
        setting = createRegressionSetting(@formula(y ~ x), dataset)
        @test jacknifedS(setting, 1) != 0
        @test jacknifedS(setting, 2) != 0
        @test jacknifedS(setting, 3) != 0
        @test jacknifedS(setting, 4) != 0
        @test jacknifedS(setting, 5) < eps
    end

    @testset "Mahalanobis squared distances" begin
        myeps = 10.0^(-6.0)
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = reverse(x)
        y[1] = y[1] * 10.0
        x[1] = x[1] * 10.0
        datamat = DataFrame(x = x, y = y)
        dmat = mahalanobisSquaredMatrix(datamat)
        d = diag(dmat)
        @test abs(d[1] - 3.2) < myeps
        @test abs(d[2] - 2.0) < myeps
        @test abs(d[3] - 0.4) < myeps
        @test abs(d[4] - 0.4) < myeps
        @test abs(d[5] - 2.0) < myeps
    end

    @testset "Find nonzero minimum" begin
        arr = [0.0, 2.0, 0.0, 0.0, 0.0, 9.0, 1.0]
        val = find_minimum_nonzero(arr)
        @test val == 1.0
    end


    @testset "covratio - phone data" begin
        eps = 0.00001
        knownvals = [
            1.2945913799871505,
            1.2700457384289985,
            1.247094858041991,
            1.2255082894941416,
            1.2056307474474428,
            1.1872216590773372,
            1.1702669047202972,
            1.1544784969566473,
            1.1415354222939595,
            1.129459077701347,
            1.1188675334881035,
            1.1092977882351849,
            1.1003495479103549,
            1.1024131168300708,
            1.0438195753346975,
            1.049139979063326,
            1.0015616430250294,
            0.9497339678816025,
            0.8530116434438035,
            0.6899325070071758,
            1.1297343653996992,
            1.0347237388802213,
            1.039659281049157,
            1.0393513929604028,
        ]
        setting = createRegressionSetting(@formula(calls ~ year), phones)
        n = length(knownvals)
        for i = 1:n
            calculated = covratio(setting, i)
            @test abs(knownvals[i] - calculated) < eps
        end
    end



    @testset "dfbeta and dfbetas - phone data" begin
        eps = 0.00001
        reg = createRegressionSetting(@formula(calls ~ year), phones)
        n, p = size(phones)
        knownvalues = [
            9.6439157 -0.14686166
            5.3459460 -0.08092134
            1.6258961 -0.02443345
            -0.6866294 0.01022725
            -2.7169197 0.04002009
            -4.1910124 0.06085238
            -5.1029254 0.07267870
            -5.5535000 0.07697356
            -5.2512176 0.06983887
            -4.6721589 0.05791933
            -3.6868718 0.03945523
            -2.3254391 0.01478033
            -0.5673087 -0.01652355
            1.4653937 -0.04958099
            -5.4474441 0.12867978
            -8.6540162 0.18101030
            -14.6631749 0.28835085
            -22.0168113 0.41708081
            -32.9443226 0.60863505
            -49.0851269 0.89065754
            22.9820710 -0.41140246
            39.1639294 -0.69370540
            45.7655562 -0.80379984
            53.6862082 -0.93638735
        ]
        for i = 1:n
            for j = 1:p
                dfbetaresult = dfbeta(reg, i)
                @test isapprox(dfbetaresult[j], knownvalues[i, j], atol = eps) 
            end
        end

        # DFBETAS
        allresults = dfbetas(reg)
        for i = 1:n
            for j = 1:p
                @test isapprox(allresults[i, j], knownvalues[i, j], atol = eps) 
            end
        end
    end



    @testset "Hadi Measure" begin
        eps = 0.0001
        setting = createRegressionSetting(@formula(calls ~ year), phones)
        knowncooks = [
            0.19101337,
            0.16141894,
            0.13677220,
            0.11673486,
            0.10058688,
            0.08815274,
            0.07913586,
            0.07353075,
            0.06965672,
            0.06906446,
            0.07108612,
            0.07605141,
            0.08428584,
            0.08612449,
            0.15005170,
            0.15622736,
            0.22082452,
            0.29817740,
            0.44310105,
            0.72642163,
            0.19942889,
            0.33018728,
            0.36907744,
            0.41937743,
        ]
        hm = hadimeasure(setting)["measure"]
        @test map((x, y) -> abs(x - y) < eps, hm, knowncooks) == trues(24)
    end


end
