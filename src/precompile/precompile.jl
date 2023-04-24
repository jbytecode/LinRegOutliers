using PrecompileTools

@setup_workload begin
    
    reg = createRegressionSetting(@formula(calls ~ year), phones)
    
    @compile_workload begin
        ols(reg)
        asm2000(reg)
        atkinson94(reg)
        bacon(reg, m = 20)
        bch(reg)
        ccf(reg)
        cm97(reg)
        hs93(reg)
        imon2005(reg)
        ks89(reg)
        lad(reg)
        lms(reg)
        lta(reg)
        lts(reg)
        quantileregression(reg)
        py95(reg)
        satman2013(reg)
        satman2015(reg)
        smr98(reg)
        ransac(reg, t = 0.8, w = 0.85)
        theilsen(reg, 2, nsamples = 10)
    end
end
