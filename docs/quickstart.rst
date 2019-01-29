.. _quickstart:

Quickstart
==========

Read in **.mesh** file and inspect mesh

::

	from dhitools import mesh
	mesh_f = "path/to/mesh/file"
	m = mesh.Mesh(mesh_f)

	 # Plotting accepts matplotlib.triplot kwargs
	kwargs = dict(color='grey', linewidth=0.8)
	f1, a1 = m.plot_mesh(kwargs=kwargs)

	f1.set_size_inches(10,10)
	a1.grid()
	a1.set_aspect('equal')
	plt.show()

.. image:: _build/html/imgs/mesh.png
    :align: center
    :alt: mesh plot

Read in **.dfsu** file and plot surface elevation at timestep 500

::

	from dhitools import dfsu
	import matplotlib.pyplot as plt

	dfsu_f = "path/to/dfsu/file"
	area = dfsu.Dfsu(dfsu_f)

	plot_dict = dict(levels = np.arange(-1,1.4,0.1))
	fig_se, ax_se, tf_se = area.plot_item(item_name='Surface elevation', tstep=400,
	                                                 kwargs=plot_dict)

	fig_se.set_size_inches(10,10)
	ax_se.set_aspect('equal')

	fig_se.colorbar(tf_se)

	ax_se.set_title('Surface elevation; t = 500')
	plt.show()

.. image:: _build/html/imgs/surface_elav.png
    :align: center
    :alt: dfsu plot