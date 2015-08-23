from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import scipy

# import AreaAvgOverTime
# import GetDataMap
#
# latRange = [20,35]
# lonRange = [75,95]
# avgOverTime1990 = AreaAvgOverTime.AreaAvgOverTime('pr', [1980,2005], latRange, lonRange)
# #avgOverTime2000 = AreaAvgOverTime.AreaAvgOverTime([2000,2005], latRange,lonRange)
# #avgOverTime = AreaAvgOverTime.AreaAvgDifference(avgOverTime1990, avgOverTime2000)
#
# tempDataLat = avgOverTime1990[0]
# tempDataLon = avgOverTime1990[1]
# tempDataData = avgOverTime1990[2]
#


flatTLat = np.array(tempDataLat)
flatTLon = np.array(tempDataLon)
flatTData = np.array(tempDataData)
print flatTData
print flatTLat
print flatTLon

m = Basemap(width=10000000/5,height=7000000/5,
            resolution='l',projection='stere',
            lat_ts = 40, lat_0=sum(latRange)/2, lon_0 = sum(lonRange)/2)

lon, lat = np.meshgrid(flatTLon[0,:], flatTLat[:,0])
x, y = m(lon,lat)
cs = m.pcolor(x,y,np.squeeze(flatTData), vmin=1e-5, vmax=5e-4)

# Add Grid Lines
m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
cbar = m.colorbar(cs, location='bottom', pad="10%")

# Add Title
plt.title('Mean Temperature in 2000-2004 Relative to 1990-1995')

plt.show()

