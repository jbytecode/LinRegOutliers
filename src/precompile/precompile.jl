using SnoopPrecompile

@precompile_setup begin
    
    reg = createRegressionSetting(@formula(calls ~ year), phones)
    
    @precompile_all_calls begin
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
        satman2013(reg)
        satman2015(reg)
        smr98(reg)
    end
end
