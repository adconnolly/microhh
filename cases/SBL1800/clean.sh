rm SBL2700.default.0000000.nc
find . -name "u.*" -not -name "*.0000000" -exec rm {} \;
find . -name "v.*" -not -name "*.0000000" -exec rm {} \;
find . -name "w.*" -not -name "*.0000000" -exec rm {} \;
find . -name "b*.*" -not -name "*.0000000" -exec rm {} \;
find . -name "p.*" -not -name "*.0000000" -exec rm {} \;
find . -name "time.*" -not -name "*.0000000" -exec rm {} \;
find . -name "d*d*.*" -not -name "*.0000000" -exec rm {} \;