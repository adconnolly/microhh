rm u.*
rm v.*
rm w.*
rm b*.*
rm p.*
rm time.*
rm d*d*.*
rm grid.*
rm fftwplan.*
#rm drycblles_input.nc
find . -name "SBL1800.*" -not -name "*.ini" -exec rm {} \;
