def CoSaMP_Model_Search(cluster_pool, correl, config_list, frozen_clusters):
	"""CoSAMP - Compressive Sampling Matching Pursuit algorithm
	Attempts to solve the original sparse-model problem, minimize_norm(A*x-b) + L0_norm(x) through a
	greedy approach. Note that LASSO replaces the L0_norm with L1_norm. This is an unconstrained minimization
	for the original noiseless-problem of min(L0_norm(x)) subject to A.x=b constraint.
	"""

	write_header("Model Selection using Compressive Sampling Matching Pursuit (COSAMP) algorithm")

	# ----Take care of frozen clusters------
	Cluster_IDList = [cls.ClusterID for cls in
					  cluster_pool]  ### Create the list of Cluster ID's from pool
	tmp1, tmp2 = [1 if b'X' in _ else 0 for _ in Cluster_IDList], [1 if cls in frozen_clusters else 0 for cls in
																   cluster_pool]	
																   #tmp1 =1 if b'X' present in clusterIDlist
																   #tmp2=1 if cls is in frozen_cluster else 0 respective to clusterpool
	Fixed_Clusters_List = list(np.array(tmp1) | np.array(tmp2))	#Bit-wise OR operation on tmp1 n tmp2 values
	# --------------------------------------

	Active_Cluster_Flags = list(np.array(Fixed_Clusters_List) | np.array([1] * len(Cluster_IDList)))	#all 1s(dbt) y not np.ones
	Feature_set = list(itertools.chain.from_iterable(
		[clsid] * len(correl[config_list[0].ID][clsid]) for (cls, clsid) in zip(Active_Cluster_Flags, Cluster_IDList) if
		cls))	#if cls of Active_Cluster is 1 then Feature_set.append(clsid*len(correl[config_list[0].ID][clsid]))
	xMatrix_Full, yDFT_Full = Create_Data_Matrix_and_Target_Vector(correl, config_list,[_ for (j, _) in enumerate(Cluster_IDList) if
																	Active_Cluster_Flags[j]])

	# ----Undersample the database----
	nDB = len(xMatrix_Full)
	if len(xMatrix_Full[0]) < len(yDFT_Full):
		_ = min(int(0.6 * len(xMatrix_Full[0])), int(0.6 * nDB))
	else:
		_ = int(0.6 * nDB)

	xMatrix, yDFT = xMatrix_Full[:_], yDFT_Full[:_]	#selecting the _th column of both
	xval, yval = xMatrix_Full[_:], yDFT_Full[_:]	#selecting the _th row of both
	# ---------------------------------

	max_allowed_sparsity = config.get('max_sparsity')	#some funtion returning max allowed sparsity
	#Valu initiallisation
	Count, Best_ECI, Best_Model, mincvs, min_train_err = 0, [], [], np.inf, 0
	upperk, lowerk, deltak = max_allowed_sparsity, 0.01, 0.01
	ksparse = upperk

	while ksparse > lowerk:
		csmp = L0greedy.CoSaMP()
		eci, train_error = csmp.Solve(yDFT, xMatrix, int(ksparse * len(xMatrix[0])), tol=1.0e-8, silent=True)
		if train_error > np.max(np.abs(np.array(
				yDFT))): break  # If magnitude of error is more than that of y - a simple check to stop runaway situation

		if eci is not None and eci != []:
			success, Active_Flags, ECIs = Select_Clusters(eci, Feature_set, Cluster_IDList, tol=1e-8)
			Model_Sparsity = float(Active_Flags.count(1)) / len(Active_Flags)	#No of 1s/total length
			#val_error =mean square value(xval.eci-yval)   err=MSE(X.A-y)
			val_error = np.sqrt(np.mean((np.dot(np.array(xval), np.array(eci)) - np.array(yval)) ** 2.0))
			if val_error < mincvs:
				Best_Model, Best_ECI, mincvs, min_train_err = Active_Flags, ECIs, val_error, train_error
				#current activeflag becomes the best model due to min mse error
			write_log("Sparsity requested: %f" % ksparse + "  Model Sparsity: %f  " % Model_Sparsity + " Train Error and Validation Error: %f   %f" % (
				train_error, val_error) + '\n')

		if abs(ksparse - deltak) < 1.0e-5:
			deltak /= 10	#value decressing like for dk=1 1,0.9,0.89,0.889,....0.88889,0.88888....,0
		ksparse -= deltak

	if Best_Model == []:
		exit_script("Cluster Selection failed with CoSaMP Algo.")#whn val_err<mincvs

	write_log(" Final Cluster-selection: " + "".join(
		str(_) for _ in Best_Model) + "  Minimum training and cvs (meV/atom):  %f  %f" % (
				  1000 * min_train_err, 1000 * mincvs) + '\n')
	return 'Converged', [[cls for (flag, cls) in zip(Best_Model, cluster_pool) if flag]], [Best_ECI], None
	#returns respective cluster_pool values for BestModel flag=1 and BestECI