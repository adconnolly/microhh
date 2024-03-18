rm u.*
rm v.*
rm w.*
rm b*.*
rm p.*
rm time.*
rm d*d*.*
rm grid.*
rm fftwplan.*
rm z0*.*
rm obuk.*
#rm drycblles_input.nc
find . -name "SBL*00.*" -not -name "*.ini" -exec rm {} \;
