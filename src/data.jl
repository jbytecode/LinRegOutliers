"""
Phone data

# Components
- `year::Integer`: years from 1950 to 1973.
- `calls::Float64`: phone calls (in millions).

# Reference
P. J. Rousseeuw and A. M. Leroy (1987) _Robust Regression &
     Outlier Detection._ Wiley.
"""
const phones = DataFrame(
 year=collect(50:73),
 calls=[4.4, 4.7, 4.7, 5.9, 6.6, 7.3, 8.1, 8.8, 10.6, 12.0, 13.5, 14.9, 16.1, 21.2, 119.0, 124.0,
          142.0, 159.0, 182.0, 212.0, 43.0, 24.0, 27.0, 29.0]
)



"""
Hawkins & Bradu & Kass data

# Components
- `x1::Float64`: first independent variable.
- `x2::Float64`: second independent variable.
- `x3::Float64`: third independent variable.
- `y::Float64`: dependent (response) variable.


# Reference
Hawkins, D.M., Bradu, D., and Kass, G.V. (1984) Location of several outliers in multiple regression data using elemental sets. Technometrics 26, 197–208.
"""
const hbk = DataFrame(
    x1=[10.1, 9.5, 10.7, 9.9, 10.3, 10.8, 10.5, 9.9, 9.7, 9.3, 11, 12, 12, 11, 3.4, 3.1, 0, 2.3, 0.8, 3.1, 2.6, 0.4, 2, 1.3, 1, 0.9, 3.3, 1.8, 1.2, 1.2, 3.1, 0.5, 1.5, 0.4, 3.1, 1.1, 0.1, 1.5, 2.1, 0.5, 3.4, 0.3, 0.1, 1.8, 1.9, 1.8, 3, 3.1, 3.1, 2.1, 2.3, 3.3, 0.3, 1.1, 0.5, 1.8, 1.8, 2.4, 1.6, 0.3, 0.4, 0.9, 1.1, 2.8, 2, 0.2, 1.6, 0.1, 2, 1, 2.2, 0.6, 0.3, 0, 0.3],
    x2=[19.6, 20.5, 20.2, 21.5, 21.1, 20.4, 20.9, 19.6, 20.7, 19.7, 24, 23, 26, 34, 2.9, 2.2, 1.6, 1.6, 2.9, 3.4, 2.2, 3.2, 2.3, 2.3, 0, 3.3, 2.5, 0.8, 0.9, 0.7, 1.4, 2.4, 3.1, 0, 2.4, 2.2, 3, 1.2, 0, 2, 1.6, 1, 3.3, 0.5, 0.1, 0.5, 0.1, 1.6, 2.5, 2.8, 1.5, 0.6, 0.4, 3, 2.4, 3.2, 0.7, 3.4, 2.1, 1.5, 3.4, 0.1, 2.7, 3, 0.7, 1.8, 2, 0, 0.6, 2.2, 2.5, 2, 1.7, 2.2, 0.4],
    x3=[28.3, 28.9, 31, 31.7, 31.1, 29.2, 29.1, 28.8, 31, 30.3, 35, 37, 34, 34, 2.1, 0.3, 0.2, 2, 1.6, 2.2, 1.9, 1.9, 0.8, 0.5, 0.4, 2.5, 2.9, 2, 0.8, 3.4, 1, 0.3, 1.5, 0.7, 3, 2.7, 2.6, 0.2, 1.2, 1.2, 2.9, 2.7, 0.9, 3.2, 0.6, 3, 0.8, 3, 1.9, 2.9, 0.4, 1.2, 3.3, 0.3, 0.9, 0.9, 0.7, 1.5, 3, 3.3, 3, 0.3, 0.2, 2.9, 2.7, 0.8, 1.2, 1.1, 0.3, 2.9, 2.3, 1.5, 2.2, 1.6, 2.6],
    y=[9.7, 10.1, 10.3, 9.5, 10, 10, 10.8, 10.3, 9.6, 9.9, -0.2, -0.4, 0.7, 0.1, -0.4, 0.6, -0.2, 0, 0.1, 0.4, 0.9, 0.3, -0.8, 0.7, -0.3, -0.8, -0.7, 0.3, 0.3, -0.3, 0, -0.4, -0.6, -0.7, 0.3, -1, -0.6, 0.9, -0.7, -0.5, -0.1, -0.7, 0.6, -0.7, -0.5, -0.4, -0.9, 0.1, 0.9, -0.4, 0.7, -0.5, 0.7, 0.7, 0, 0.1, 0.7, -0.1, -0.3, -0.9, -0.3, 0.6, -0.3, -0.5, 0.6, -0.9, -0.7, 0.6, 0.2, 0.7, 0.2, -0.2, 0.4, -0.9, 0.2]
)



"""
Animals data

# Components
- `names::AbstractString`: names of animals.
- `body::Float64`: body weight in kg.
- `brain::Float64`: brain weight in g.


# References
     Venables, W. N. and Ripley, B. D. (1999) _Modern Applied
     Statistics with S-PLUS._ Third Edition. Springer.

     P. J. Rousseeuw and A. M. Leroy (1987) _Robust Regression and
     Outlier Detection._ Wiley, p. 57.
"""
const animals = DataFrame(
    names=["Mountain beaver", "Cow", "Grey wolf", "Goat", "Guinea pig", "Dipliodocus", "Asian elephant", "Donkey", "Horse", "Potar monkey", "Cat", "Giraffe", "Gorilla", "Human", "African elephant", "Triceratops", "Rhesus monkey", "Kangaroo", "Golden hamster", "Mouse", "Rabbit", "Sheep", "Jaguar", "Chimpanzee", "Rat", "Brachiosaurus", "Mole", "Pig"],
    body=[1.35, 465, 36.33, 27.66, 1.04, 11700, 2547, 187.1, 521, 10, 3.3, 529, 207, 62, 6654, 9400, 6.8, 35, 0.12, 0.023, 2.5, 55.5, 100, 52.16, 0.28, 87000, 0.122, 192],
    brain=[8.1, 423, 119.5, 115, 5.5, 50, 4603, 419, 655, 115, 25.6, 680, 406, 1320, 5712, 70, 179, 56, 1, 0.4, 12.1, 175, 157, 440, 1.9, 154.5, 3, 180]
)



"""
Weight loss data

# Components
- `days::Integer`: time in days since the start of the diet program.
- `weight::Float64`: weight in kg.


# Reference
     Venables, W. N. and Ripley, B. D. (1999) _Modern Applied
     Statistics with S-PLUS._ Third Edition. Springer.
"""
const weightloss = DataFrame(
    days=[0, 4, 7, 7, 11, 18, 24, 30, 32, 43, 46, 60, 64, 70, 71, 71, 73, 74, 84, 88, 95, 102, 106, 109, 115, 122, 133, 137, 140, 143, 147, 148, 149, 150, 153, 156, 161, 164, 165, 165, 170, 176, 179, 198, 214, 218, 221, 225, 233, 238, 241, 246],
    weight=[184.35, 182.51, 180.45, 179.91, 177.91, 175.81, 173.11, 170.06, 169.31, 165.1, 163.11, 158.3, 155.8, 154.31, 153.86, 154.2, 152.2, 152.8, 150.3, 147.8, 146.1, 145.6, 142.5, 142.3, 139.4, 137.9, 133.7, 133.7, 133.3, 131.2, 133, 132.2, 130.8, 131.3, 129, 127.9, 126.9, 127.7, 129.5, 128.4, 125.4, 124.9, 124.9, 118.2, 118.2, 115.3, 115.7, 116, 115.5, 112.6, 114, 112.6]
)


"""
Stack loss data

# Components
- `airflow::Float64`: flow of cooling air (independent variable).
- `watertemp::Float64`: cooling water inlet temperature (independent variable).
- `acidcond::Float64`: concentration of acid (independent variable).
- `stackloss::Float64`: stack loss (dependent variable).

# Outliers
    Observations 1, 3, 4, and 21 are outliers.

# References
    Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988) _The New S Language_.  Wadsworth & Brooks/Cole.

    Dodge, Y. (1996) The guinea pig of multiple regression. In: _Robust Statistics, Data Analysis, and Computer Intensive Methods;
    In Honor of Peter Huber's 60th Birthday_, 1996, _Lecture Notes in Statistics_ *109*, Springer-Verlag, New York.
"""
const stackloss = DataFrame(
    airflow=[80.0, 80, 75, 62, 62, 62, 62, 62, 58, 58, 58, 58, 58, 58, 50, 50, 50, 50, 50, 56, 70],
    watertemp=[27.0, 27, 25, 24, 22, 23, 24, 24, 23, 18, 18, 17, 18, 19, 18, 18, 19, 19, 20, 20, 20],
    acidcond=[89.0, 88, 90, 87, 87, 87, 93, 93, 87, 80, 89, 88, 82, 93, 89, 86, 72, 79, 80, 82, 91],
    stackloss=[42.0, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8, 7, 8, 8, 9, 15, 15]
)



"""
Hadi & Simonoff (1993) Random data

# Components
- `x1::Float64`: Random values.
- `x2::Float64`: Random values.
- `y::Float64`: Random values (independent variable).

# Outliers
    Observations 1, 2, and 3 are outliers.

# References
Hadi, Ali S., and Jeffrey S. Simonoff. "Procedures for the identification of 
multiple outliers in linear models." Journal of the American Statistical 
Association 88.424 (1993): 1264-1272.
"""
const hs93randomdata = DataFrame(
    x1=[15.0, 14.95, 14.9, 13.7909, 3.60424, 6.3905, 5.13756, 6.98242, 8.54968, 9.41802, 8.58942, 2.32221, 14.3409, 8.08558, 9.38206, 4.32664, 12.7987, 7.92987, 2.26946, 12.8386, 3.66188, 0.254591, 3.91701, 13.1687, 0.953868],
    x2=[15.0, 14.95, 14.9, 17.0999, 3.00793, 11.0249, 9.11747, 6.70008, 6.58946, 7.14941, 3.39223, -0.115934, 10.935, 5.28895, 2.28385, 3.50448, 15.7696, 13.8133, 6.30972, 5.66232, 3.55028, 0.230256, 11.5185, 9.82886, 4.29971],
    y=[34.0, 33.9, 33.8, 32.6016, 6.38553, 16.1119, 14.8969, 14.0065, 15.1297, 17.9406, 10.22806, 1.6473, 25.1874, 13.764, 11.9967, 8.2127, 28.0132, 22.8697, 7.15197, 18.2589, 7.09652, 0.17605, 15.8921, 22.4458, 4.65196]
)



"""
Modified Wood Gravity Data

# Components
- `x1::Float64`: Random values.
- `x2::Float64`: Random values.
- `x3::Float64`: Random values.
- `x4::Float64`: Random values.
- `x5::Float64`: Random values.
- `y::Float64`: Random values (independent variable).


# References
P. J. Rousseeuw and A. M. Leroy (1987) Robust Regression and Outlier Detection. Wiley, p.243, table 8.
"""
const woodgravity = DataFrame(
    x1=[0.573, 0.651, 0.606, 0.437, 0.547, 0.444, 0.489, 0.413, 0.536, 0.685, 0.664, 0.703, 0.653, 0.586, 0.534, 0.523, 0.58, 0.448, 0.417, 0.528],
    x2=[0.1059, 0.1356, 0.1273, 0.1591, 0.1135, 0.1628, 0.1231, 0.1673, 0.1182, 0.1564, 0.1588, 0.1335, 0.1395, 0.1114, 0.1143, 0.132, 0.1249, 0.1028, 0.1687, 0.1057],
    x3=[0.465, 0.527, 0.494, 0.446, 0.531, 0.429, 0.562, 0.418, 0.592, 0.631, 0.506, 0.519, 0.625, 0.505, 0.521, 0.505, 0.546, 0.522, 0.405, 0.424],
    x4=[0.538, 0.545, 0.521, 0.423, 0.519, 0.411, 0.455, 0.43, 0.464, 0.564, 0.481, 0.484, 0.519, 0.565, 0.57, 0.612, 0.608, 0.534, 0.415, 0.566],
    x5=[0.841, 0.887, 0.92, 0.992, 0.915, 0.984, 0.824, 0.978, 0.854, 0.914, 0.867, 0.812, 0.892, 0.889, 0.889, 0.919, 0.954, 0.918, 0.981, 0.909],
    y=[0.534, 0.535, 0.57, 0.45, 0.548, 0.431, 0.481, 0.423, 0.475, 0.486, 0.554, 0.519, 0.492, 0.517, 0.502, 0.508, 0.52, 0.506, 0.401, 0.568]
)




"""
Scottish Hill Races Data

# Components
- `dist::Array{Float64, 1}`: Distance in miles (Independent). 
- `climb::Array{Float64, 1}`: Heights in feet (Independent).
- `time::Array{Float64, 1}`: Record times in hours (Dependent).

# Model 
time ~ dist + climb

# References
A.C. Atkinson (1986) Comment: Aspects of diagnostic regression analysis. Statistical Science 1, 397-402.
"""
const hills = DataFrame(
    dist=[2.4, 6.0, 6.0, 7.5, 8.0, 8.0, 16.0, 6.0, 5.0, 6.0, 28.0, 5.0, 9.5, 6.0, 4.5, 10.0, 14.0, 3.0, 4.5, 5.5, 3.0, 3.5, 6.0, 2.0, 3.0, 4.0, 6.0, 5.0, 6.5, 5.0, 10.0, 6.0, 18.0, 4.5, 20.0],
    climb=[650.0, 2500.0, 900.0, 800.0, 3070.0, 2866.0, 7500.0, 800.0, 800.0, 650.0, 2100.0, 2000.0, 2200.0, 500.0, 1500.0, 3000.0, 2200.0, 350.0, 1000.0, 600.0, 300.0, 1500.0, 2200.0, 900.0, 600.0, 2000.0, 800.0, 950.0, 1750.0, 500.0, 4400.0, 600.0, 5200.0, 850.0, 5000.0],
    time=[0.268055555555556, 0.805833333333333, 0.560833333333333, 0.76, 1.03777777777778, 1.22027777777778, 3.41027777777778, 0.606111111111111, 0.495833333333333, 0.6625, 3.21111111111111, 0.7175, 1.08333333333333, 0.735555555555556, 0.448888888888889, 1.20416666666667, 1.64027777777778, 1.31083333333333, 0.290277777777778, 0.542777777777778, 0.265833333333333, 0.465, 0.794166666666667, 0.298888888888889, 0.311388888888889, 0.436944444444444, 0.573888888888889, 0.476111111111111, 0.841666666666667, 0.349166666666667, 1.42638888888889, 0.539722222222222, 2.8375, 0.468333333333333, 2.66388888888889]
)