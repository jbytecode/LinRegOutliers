@testset "MVE - Algorithm - Phone data" begin
    df = phones
    outset = mve(df)["outliers"]
    @test 15 in outset
    @test 16 in outset
    @test 17 in outset
    @test 18 in outset
    @test 19 in outset
    @test 20 in outset
    @test 21 in outset
end

@testset "MCD - Algorithm - Phone data" begin
    df = phones
    outset = mcd(df)["outliers"]
    @test 15 in outset
    @test 16 in outset
    @test 17 in outset
    @test 18 in outset
    @test 19 in outset
    @test 20 in outset
    @test 21 in outset
end


@testset "MVE & LTS Plot - Algorithm - Phone data" begin
    df = phones
    reg = createRegressionSetting(@formula(calls ~ year), df)
    result = mveltsplot(reg, showplot=false)
    regulars = result["regular.points"]
    @test 1 in regulars
    @test 2 in regulars
    @test 3 in regulars
    @test 4 in regulars
    @test 5 in regulars
    @test 6 in regulars
    @test 7 in regulars
    @test 8 in regulars
    @test 9 in regulars
    @test 10 in regulars
    @test 11 in regulars
    @test 12 in regulars
    @test 13 in regulars
end
