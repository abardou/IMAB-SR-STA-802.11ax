import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math as m
import seaborn as sns
from matplotlib import lines
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import proj3d, art3d
from mpl_toolkits.mplot3d.art3d import Line3D
import re
from matplotlib.patches import Ellipse, ConnectionPatch, Circle
import json

# sns.set_theme()
# sns.set(font_scale=1.8)

markers_list = ["o", "^", "s", "P", "d", "*"]
linewidth = 3
markersize = 12

plt.rcParams.update({'font.size': 22})

alpha = 0.04
sample = 90

mc = 3000
ncol = 1

def prepareDf(df):
	colsToDrop = [c for c in df if int(c) > mc]
	
	return df.drop(colsToDrop, axis=1)

def prepareDfs(dfs):
	l = []
	for df in dfs:
		l.append(prepareDf(df))

	return l

def distance(xA, yA, xB, yB):
	return m.sqrt((xA - xB) ** 2 + (yA - yB) ** 2)

def receivingPower(tx, p1, p2):
	return tx - 46.67 - 10 * 3 * m.log10(distance(*p1, *p2)) # 3.24


def stringToConfiguration(confString):
	return eval('['+confString+']')

def extractAPsConflicts(topo, configuration, pathLoss, channels=None, args=[]):
	conflicts = [[False for _ in range(len(topo['aps']))] for _ in range(len(topo['aps']))]
	degs = []
	for i,emitting in enumerate(topo['aps']):
		deg = 0
		for j,receiving in enumerate(topo['aps']):
			if i != j and (channels is None or channels[i] == channels[j]):
				conflicts[i][j] = pathLoss(configuration[i][1], (emitting['x'], emitting['y']), (receiving['x'], receiving['y']), *args) >= configuration[j][0]
				if conflicts[i][j]:
					deg += 1

		degs.append(deg)

	print(degs)
	print(sum(degs))
	print("Average degree:", round(sum(degs) / len(degs), 2))

	return conflicts

def maxAPStaDistance(topoFile):
	topo = json.load(open(topoFile))
	dmaxs = []
	for ap in topo['aps']:
		dmax = 0
		for staid in ap['stas']:
			challenger = distance(ap['x'], ap['y'], topo['stations'][staid]['x'], topo['stations'][staid]['y'])
			if challenger > dmax:
				dmax = challenger
		
		dmaxs.append(dmax)

	return dmaxs

def plot2dTopology(topoFile, topoName, pathLoss, configuration=None, args=[]):
	"""Show a topology with matplotlib
	Arguments:
			aps {list} -- list of AccessPoint to show
			stations {list} -- list of Stations to show
	"""
	topo = json.load(open(topoFile))
	# Size of elements and range
	ap_rad = 1.5
	sta_rad = ap_rad / m.sqrt(2)
	shift = 10
	min_x, min_y = 1e10, 1e10
	max_x, max_y = -1e10, -1e10
	for o in topo['aps'] + topo['stations']:
		if o['x'] > max_x:
			max_x = o['x']
		if o['x'] < min_x:
			min_x = o['x']
		if o['y'] > max_y:
			max_y = o['y']
		if o['y'] < min_y:
			min_y = o['y']
	# Graphic construction
	_, ax = plt.subplots(figsize=(12, 9))
	ax.set_xlim(left=min_x-shift, right=max_x+shift)
	ax.set_ylim(bottom=min_y-shift, top=max_y+shift)
	dx = max_x-min_x+2*shift
	dy = max_y-min_y+2*shift
	frx = dx / dy
	# Elements : access points, stations, connections
	ap_circles = [Ellipse((ap['x'], ap['y']), width=2*frx*ap_rad, height=2*ap_rad, color='r', zorder=1) for ap in topo['aps']]
	sta_circles = [Ellipse((sta['x'], sta['y']), width=2*frx*sta_rad, height=2*sta_rad, color='b', zorder=0) for sta in topo['stations']]
	ap_sta_lines = []
	for ap in topo['aps']:
		ap_sta_lines += [plt.Line2D([topo['stations'][sta]['x'], ap['x']], [topo['stations'][sta]['y'], ap['y']], color='gray', linewidth=0.15, zorder=-1) for sta in ap['stas']]
	if configuration is None:
		configuration = [(-82,20) for _ in range(len(topo["aps"]))]
	conflicts = extractAPsConflicts(topo, configuration, pathLoss, args=args)
	ap_lines = [ConnectionPatch((topo['aps'][i]['x'], topo['aps'][i]['y']), (topo['aps'][j]['x'], topo['aps'][j]['y']), "data", "data", arrowstyle="-|>", shrinkA=7*ap_rad, shrinkB=7*ap_rad, mutation_scale=40, fc="black") for i in range(len(topo['aps'])) for j in range(len(topo['aps'])) if conflicts[i][j]]
	for c in ap_lines:
		ax.add_artist(c)
	for c in ap_circles:
		ax.add_artist(c)
	for c in sta_circles:
		ax.add_artist(c)
	for l in ap_sta_lines:
		ax.add_artist(l)
	for i,sta in enumerate(topo['stations']):
		plt.text(sta['x'] + 1.2*frx*sta_rad, sta['y'] + 1.2*sta_rad, str(i+1), color='b', fontsize=16)
	for i,ap in enumerate(topo['aps']):
		plt.text(ap['x'] + 1.2*frx*ap_rad, ap['y'] + 1.2*ap_rad, str(i+1), color='r', fontsize=21)
	ax.set_xlabel("x (meters)"); ax.set_ylabel("y (meters)")
	# plt.title(topoName + ": " + str(configuration))
	plt.show()

def find_combined_max(dfs):
	maxs = []
	for df in dfs:
		maxs.append(df.max().max())

	return max(maxs)

def getRegrets(files):
	dfs = prepareDfs([pd.read_csv(f, sep="\t", index_col=False) for f in files])
	xxs = [range(len(df.iloc[0])) for df in dfs]

	m = 1 # find_combined_max(dfs)

	# Replace reward with regret and compute the curves
	for i in range(len(dfs)):
		for line,row in dfs[i].iterrows():
			dfs[i].iloc[line] = m - row

	return dfs

def plotRegrets(files, names, topoName, legend=True):
	dfs = getRegrets(files)
	xxs = [range(len(df.iloc[0])) for df in dfs]

	stats = []
	for i in range(len(dfs)):
		for line,_ in dfs[i].iterrows():
			dfs[i].loc[line] = dfs[i].loc[line].ewm(alpha=alpha).mean()

		low = []
		mid = []
		up = []
		for col in dfs[i]:
			quantiles = dfs[i][col].quantile([0.25, 0.5, 0.75])
			low.append(quantiles[0.25])
			mid.append(quantiles[0.5])
			up.append(quantiles[0.75])

		stats.append((low,mid,up))
	
	fig, ax = plt.subplots(figsize=(11, 8))
	for i,s in enumerate(stats):
		low,mid,up = s
		print(len(xxs), len(names))
		p = ax.plot(xxs[i], mid, label=names[i]) # '-', '--', '-.', ':', 'None', ' ', ''
		ax.fill_between(xxs[i], low, up, color=p[0].get_color(), alpha=0.15)

	# plt.title(topoName + ': Regret evolution for each strategy')
	plt.xlabel("Optimization steps")
	plt.ylabel("Regret")
	if legend:
		plt.legend(ncol=ncol)
	plt.show()

def plotCumRegrets(files, names, topoName, legend=True):
	dfs = getRegrets(files)
	xxs = [range(len(df.iloc[0])) for df in dfs]

	stats = []
	for i in range(len(dfs)):
		for line,_ in dfs[i].iterrows():
			dfs[i].loc[line] = dfs[i].loc[line].cumsum()

		low = []
		mid = []
		up = []
		for col in dfs[i]:
			quantiles = dfs[i][col].quantile([0.25, 0.5, 0.75])
			low.append(quantiles[0.25])
			mid.append(quantiles[0.5])
			up.append(quantiles[0.75])

		stats.append((low,mid,up))
	
	fig, ax = plt.subplots(figsize=(11, 8))
	for i,s in enumerate(stats):
		low,mid,up = s
		p = ax.plot(xxs[i], mid, label=names[i])
		ax.fill_between(xxs[i], low, up, color=p[0].get_color(), alpha=0.15)

	# plt.title(topoName + ': Regret evolution for each strategy')
	plt.xlabel("Optimization steps")
	plt.ylabel("Cumulative Regret")
	if legend:
		plt.legend(ncol=ncol)
	plt.show()

def extractConfigReward(confs, dfr, config, mod=False):
	rew = []
	indexes = extractIndexes(confs, config)
	for r,rindexes in enumerate(indexes):
		for idx in rindexes:
			rew.append(dfr.iloc[r][str(idx)])

	me = np.mean(rew)

	return (len(rew) * me + 1) / (len(rew) + 2) if mod else me

def bestConfigurations(files, rewardFiles):
	
	dfs = prepareDfs([pd.read_csv(f, sep="\t", index_col=False) for f in files])
	best = []
	for i,df in enumerate(dfs):
		df_best = pd.Series()
		rewards = {}
		confs = pd.read_csv(files[i], sep='\t', index_col=False)
		dfr = pd.read_csv(rewardFiles[i], sep='\t', index_col=False)
		for k,row in df.iterrows():
			best_conf = None
			best_rew = -1
			for conf in pd.unique(row):
				if conf not in rewards:
					rewards[conf] = extractConfigReward(confs, dfr, conf)
				chall = rewards[conf]
				# print(best_rew, chall, conf)
				if chall > best_rew:
					best_rew = chall
					best_conf = conf

			df_best = df_best.append(pd.Series(best_conf), ignore_index=True)
			rewards[best_conf] = best_rew

		bestConfigs = df_best.value_counts()
		rewards = [rewards[c] for c in bestConfigs.index]

		bestConfigsv = [x for _,x in sorted(zip(rewards, bestConfigs.values), reverse=True)]
		bestConfigsi = [x for _,x in sorted(zip(rewards, bestConfigs.index), reverse=True)]
		bestConfigs = pd.DataFrame(data=bestConfigsv, index=bestConfigsi)
		best.append(bestConfigs)

		print("OK")
	
	return best

def plotBestConfigurations(files, sta_files, names, topoName, topoFile, rewardFiles):
	bestConfigs = bestConfigurations(files, rewardFiles)
	for i in range(len(bestConfigs)):
		# Extract average reward
		rewards = []
		df_conf = pd.read_csv(files[i], sep='\t', index_col=False)
		dfr = pd.read_csv(rewardFiles[i], sep='\t', index_col=False)
		print(bestConfigs[i][0].index[0].replace('.000000', ''))
		for j in range(len(bestConfigs[i])):
			rewards.append(round(extractConfigReward(df_conf, dfr, bestConfigs[i].index[j]), 3))

		lx = bestConfigs[i].index.to_series().apply(lambda x: x.replace('.000000', ''))
		y = np.reshape(bestConfigs[i].values, -1)

		print(y)

		fig, ax = plt.subplots(figsize=(16, 9))
		ind = np.arange(len(y))  # the x locations for the groups
		ax.barh(ind, y)

		for j in range(len(y)):
			ax.text(y[j] + 0.25, j, rewards[j])

		ax.set_yticks(ind)
		ax.set_yticklabels(lx)
		ax.set_xlim(0, sum(y))
		# plt.title(topoName + ": Best configurations for " + names[i] + " (Pol: " + str(pol) + "%)")
		plt.xlabel("# Use")
		plt.ylabel("Configurations")      

		plt.tight_layout()
		plt.show()
		# plt.savefig(os.path.join('test.png'), dpi=300, format='png', bbox_inches='tight')

		# Topology with this configuration
		plot2dTopology(topoFile, topoName, receivingPower, configuration=eval(f"[{bestConfigs[i][0].index[0].replace('.000000', '')}]"))

def extractIndexes(df, value):
	indexes = []
	for _,row in df.iterrows():
		row_indexes = []
		for i,v in enumerate(row.values):
			if v == value:
				row_indexes.append(i)
		indexes.append(row_indexes)

	return indexes

def plotStarvations(files, names, topoFile, attainableThroughput, topoName, legend=True):
	topo = json.load(open(topoFile))['aps']
	aThroughputs = []
	xxs = []

	for ap in topo:
		for _ in ap['stas']:
			aThroughputs.append(0.1 * attainableThroughput / len(ap['stas']))

	stats = []
	for i,file in enumerate(files):
		low = []
		up = []
		mid = []

		df = prepareDf(pd.read_csv(file, sep="\t", index_col=False))
		xxs.append(range(len(df.iloc[0])))
		for c in df:
			for j,_ in df.iterrows():
				df[c].iloc[j] = sum([float(x) < aThroughputs[k] for k,x in enumerate(df[c].iloc[j].split(','))])

			quantiles = df[c].quantile([0.25, 0.5, 0.75])
			low.append(quantiles[0.25])
			mid.append(quantiles[0.5])
			up.append(quantiles[0.75])
		
		stats.append((low,mid,up))

	fig, ax = plt.subplots(figsize=(11, 8))
	for i,s in enumerate(stats):
		low,mid,up = s
		p = ax.plot(xxs[i], pd.Series(mid).ewm(alpha=alpha).mean(), label=names[i])
		ax.fill_between(xxs[i], pd.Series(low).ewm(alpha=alpha).mean(), pd.Series(up).ewm(alpha=alpha).mean(), color=p[0].get_color(), alpha=0.15)

	# plt.title(topoName + ': ' + "Starvations")
	plt.xlabel("Optimization steps")
	plt.ylabel("Number of starvations")
	if legend:
		plt.legend(ncol=ncol)
	plt.show()

def plotSearchScalars(files, names, ylab, title, topoName, factor=1, legend=True, factors=False, log=False):
	fact = [factor * f for f in factors] if type(factors) == list else [factor for _ in range(len(files))]
	print(fact)
	dfs = prepareDfs([pd.read_csv(f, sep="\t", index_col=False) / fact[i] for i,f in enumerate(files)])
	xxs = [range(0, len(df.iloc[0]), sample) for df in dfs]

	# Replace reward with regret and compute the curves
	stats = []
	for i in range(len(dfs)):
		low = []
		up = []
		mid = []

		for k,col in enumerate(dfs[i]):
			if k % sample == 0:
				quantiles = dfs[i][col].quantile([0.25, 0.5, 0.75])
				low.append(quantiles[0.25])
				mid.append(quantiles[0.5])
				up.append(quantiles[0.75])

		mean, std = dfs[i][dfs[i].columns[-1]].mean(), dfs[i][dfs[i].columns[-1]].std()
		print(f"CI of {names[i]} at the end: {round(mean, 2)} +/- {round(1.96 * std / m.sqrt(dfs[i].shape[0]), 2)}")

		stats.append((low,mid,up))
	
	fig, ax = plt.subplots(figsize=(11, 8))
	for i,s in enumerate(stats):
		low,mid,up = s
		p = ax.plot(xxs[i], pd.Series(mid).ewm(alpha=alpha).mean(), label=names[i], marker=markers_list[i], linewidth=linewidth, markersize=markersize)
		# ax.fill_between(xxs[i], pd.Series(low).ewm(alpha=alpha).mean(), pd.Series(up).ewm(alpha=alpha).mean(), color=p[0].get_color(), alpha=0.15)

	# plt.title(topoName + ': ' + title)
	if log:
		plt.yscale("log")
	plt.xlabel("Optimization steps")
	plt.ylabel(ylab)
	if legend:
		plt.legend(ncol=ncol)
	plt.show()

def compareScalars(confFiles, rewardFiles, files, names, xlab, ylab, title, topoName):
	bestConfigs = bestConfigurations(confFiles, rewardFiles)
	bestConfigs = [configs.index[0] for configs in bestConfigs]

	scalars = []
	for i,config in enumerate(bestConfigs):
		file_scalars = []

		indexes = extractIndexes(pd.read_csv(confFiles[i], sep='\t', index_col=False), config)
		df = prepareDf(pd.read_csv(files[i], sep="\t", index_col=False))
		for r,rindexes in enumerate(indexes):
			for c in rindexes:
				file_scalars.append(df[str(c)].iloc[r])
		
		scalars.append(file_scalars)

	scalars = np.array(scalars, dtype=object)
	
	fig,ax = plt.subplots(figsize=(16, 9))
	ax.boxplot(scalars, labels=names, notch=True)

	plt.xlabel(xlab)
	plt.ylabel(ylab)
	# plt.title(topoName + ': ' + title)
	plt.show()

def plotAPsThroughputs(confFiles, rewardFiles, files, names, topoName):
	bestConfigs = bestConfigurations(confFiles, rewardFiles)
	bestConfigs = [configs.index[0] for configs in bestConfigs]

	throughputs = []
	for i,config in enumerate(bestConfigs):
		file_throughputs = []
		indexes = extractIndexes(pd.read_csv(confFiles[i], sep='\t', index_col=False), config)
		df = prepareDf(pd.read_csv(files[i], sep="\t", index_col=False))
		for r,rindexes in enumerate(indexes):
			for c in rindexes:
				file_throughputs.append(df[str(c)].iloc[r])

		# Cast file_throughputs in array of throughputs
		file_throughputs = np.array([[float(x) for x in t.split(',')] for t in file_throughputs])	
		throughputs.append(file_throughputs)

	for i,t in enumerate(throughputs):
		fig,ax = plt.subplots(figsize=(16, 9))
		ax.boxplot(t, labels=["AP " + str(i+1) for i in range(len(t[0]))], notch=True)

		plt.xlabel("APs")
		plt.ylabel("Throughput")
		# plt.title(topoName + ": APs throughputs distribution for " + names[i])
		plt.show()

def plotTxVsObss():
	def upBound(tx):
		return max(-82, min(-62, -82 + 20 - tx))
	
	xx = np.linspace(0, 22, 1000)
	yy = []
	for x in xx:
		yy.append(upBound(x))
	
	_, ax = plt.subplots(figsize=(12, 8))
	ax.plot(xx, yy)
	ax.fill_between(xx, yy, [-82]*1000, alpha=0.2, label="Authorized")
	# ax.set_yticks(np.arange(-82, -61, 2))
	ax.set_xlabel("TX_POWER (dBm)")
	ax.set_ylabel("OBSS/PD (dBm)")
	plt.legend(ncol=ncol)
	plt.show()

def averageSTAAPDistance(topology):
	t = json.load(open(topology))
	aps, stas = t['aps'], t['stations']
	
	dist = 0
	n = 0
	for ap in aps:
		x,y = ap['x'], ap['y']
		for sta in ap['stas']:
			xs,ys = stas[sta]['x'], stas[sta]['y']
			dist += m.sqrt((x - xs) ** 2 + (y - ys) ** 2)
			n += 1

	return dist / n

def averageSTAInterfAPDistance(topology):
	t = json.load(open(topology))
	aps, stas = t['aps'], t['stations']
	
	avg_dist = 0
	n = 0
	for ap in aps:
		for sta in ap['stas']:
			xs,ys = stas[sta]['x'], stas[sta]['y']
			min_dist = 10000000
			for interf_ap in aps:
				if interf_ap != ap:
					xa,ya = interf_ap['x'], interf_ap['y']
					dist = m.sqrt((xs - xa) ** 2 + (ys - ya) ** 2)
					if dist < min_dist:
						min_dist = dist
			n += 1
			avg_dist += min_dist

	return avg_dist / n

def plotRewardFunctionCut(path):
	data = pd.read_csv(path, sep="\t", index_col=False)
	pd.set_option('display.max_rows', None)
	print(data)
	print(data.loc[data['rew'].idxmax()])
	rewards = {}
	for i in range(len(data)):
		rewards[(data['x'].iloc[i], data['y'].iloc[i])] = data['rew'].iloc[i]

	xx, yy = pd.unique(data['x']), pd.unique(data['y'])
	X, Y = np.meshgrid(xx, yy)
	Z, Zmin, Zmax = [], [], []

	zmax = 0
	zmax_x = None
	zmax_y = None
	for i,xline in enumerate(X):
		zline = []
		zlinemin = []
		zlinemax = []
		for j,x in enumerate(xline):
			z = rewards[(x, Y[i, j])] if (x, Y[i, j]) in rewards else np.nan
			zline.append(z)
			zlinemin.append(rewards[(x, Y[i, j])] if (x, Y[i, j]) in rewards else 1)
			zlinemax.append(rewards[(x, Y[i, j])] if (x, Y[i, j]) in rewards else 0)

			if zmax_x is None or (not np.isnan(z) and zmax < z):
				zmax = z
				zmax_x = x
				zmax_y = Y[i, j]

		Z.append(zline)
		Zmax.append(zlinemax)
		Zmin.append(zlinemin)
	
	Z = np.array(Z)
	Zmin = np.array(Zmin)
	Zmax = np.array(Zmax)

	vmin = np.min(Zmin)
	vmax = np.max(Zmax)

	print(zmax, rewards[(0, 0)], round(100 * (zmax - rewards[(0, 0)]) / rewards[(0, 0)], 2))

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	# ax.scatter([zmax_x], [zmax_y], [zmax + 0.05 * (vmax - vmin)], c='red', s=40, marker='v')
	surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, vmin=vmin, vmax=vmax, linewidth=0.1, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)

	p = Circle((0, 0), 0.5, ec='red', fc="none", lw=2)
	ax.add_patch(p)
	art3d.pathpatch_2d_to_3d(p, z=zmax - 0.02 * (vmax - vmin), zdir="z")
	plt.show()


# plotRewardFunctionCut("./data/MER_FLOORS_CH20_S5_RandomBasis_results.tsv")

duration = 120.0
testDurations = [0.05]

topos = ["C6o", "T12"]
tests = ["DEFAULT_UNI_ADHOC", "TGNORM_HGMT_ADHOC"] # "DEFAULT_UNI_ADHOC", "EGREED_UNI_ADHOC", "TNORM_UNI_ADHOC", "TNORM_HGMT_ADHOC", 
# saturation = [0.0, 0.333333, 0.666667, 1.0]

for topo in topos:
	for testDuration in testDurations:
		templates = ['data/' + topo + '_' + str(duration) + '_' + t + '_' + str(testDuration) for t in tests]
		# names = ["BEST 0.7"]
		names = ["Legacy configuration", "SR improvement technique"] # "DEFAULT", "ε-GREEDY", "TS", "GM-TS", 
		topology = 'topos/'+topo+".json"
		print(topology)
		print("Average STA-AP distance:", averageSTAAPDistance(topology))
		print("Average STA-InterfAP distance:", averageSTAInterfAPDistance(topology))
		print("Ratio Metric:", averageSTAInterfAPDistance(topology) / averageSTAAPDistance(topology))

		# Plot the topology
		plot2dTopology(topology, topo, receivingPower)
		# Plot the regret
		# plotRegrets([t+"_rew.tsv" for t in templates], names, topo, legend=True)
		# # # # # # Plot the cumulative regret for all the files
		# plotCumRegrets([t+"_rew.tsv" for t in templates], names, topo)
		# # # # # Throughputs des stations en fonction du temps
		# plotSearchScalars([t+"_rew.tsv" for t in templates], names, "Reward", "Reward during the search", topo, legend=True)
		# # Compute and plot starvations during search
		# plotStarvations([t+"_stas.tsv" for t in templates], names, topology, 600e6, topo, legend=True)
		# # # # # # Compute and plot fairness during search
		# plotSearchScalars([t+"_fair.tsv" for t in templates], names, "Fairness",
		# 		"Fairness during the search", topo, legend=True)
		# # # # # Compute and plot fairness during search
		plotSearchScalars([t+"_cum.tsv" for t in templates], names, "Aggregate Throughput (Mbps)",
			"Aggregate Throughput during the search", topo, 1e6, legend=True)
		# # Extract and print the best configuration for all the files
		plotBestConfigurations([t+"_conf.tsv" for t in templates], [t+"_stas.tsv" for t in templates], names, topo, topology, [t+"_rew.tsv" for t in templates])
		# # # Compute and plot fairness to compare between simulations
		# compareScalars([t+"_conf.tsv" for t in templates], [t+"_rew.tsv" for t in templates], [t+"_fair.tsv" for t in templates], names, "Strategies", "Fairness",
		# 	"Fairness distribution for each best configuration found", topo)
		# # Compute and plot cumulated throughput to compare between simulations
		# compareScalars([t+"_conf.tsv" for t in templates], [t+"_rew.tsv" for t in templates], [t+"_cum.tsv" for t in templates], names, "Strategies", "Cumulated Throughput",
		# 	"Cumulated throughput distribution for each best configuration found", topo)
		# Compare APs throughputs
		# plotAPsThroughputs([t+"_conf.tsv" for t in templates], [t+"_rew.tsv" for t in templates], [t+"_aps.tsv" for t in templates], names, topo)
		# # Compare STAs throughputs
		# plotSTAsThroughputs([t+"_conf.tsv" for t in templates], [t+"_rew.tsv" for t in templates], [t+"_stas.tsv" for t in templates], names, topology, 300e6, topo)