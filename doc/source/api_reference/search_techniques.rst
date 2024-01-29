Search Techniques
=================

.. py:data:: Coordinates: Tuple[float, ...]

  Coordinates are in pyATF represented as a tuple of :code:`float` s.

.. py:data:: Cost: float

  We currently use :code:`float` as cost type.

.. py:class:: pyatf.search_techniques.search_technique.SearchTechnique

  Searches over multi-dimensional coordinate space :math:`(0,1]^D`.

  .. py:function:: initialize(dimensionality: int)

    Initializes the search technique.

    :param dimensionality: "D" of the coordinate space

  .. py:function:: finalize()

    Finalizes the search technique.

  .. py:function:: get_next_coordinates() -> Set[Coordinates]

    Returns the next coordinates in :math:`(0,1]^D` for which the costs are requested.

    Function :code:`get_next_coordinates()` is called by pyATF before each call to :code:`report_costs(...)`.

    :return: coordinates in :math:`(0,1]^D`

  .. py:function:: report_costs(costs: Dict[Coordinates, Cost])

    Processes costs for coordinates requested via function :code:`get_next_coordinates()`.

    Function :code:`report_costs(...)` is called by pyATF after each call to :code:`get_next_coordinates()`.

    :param costs: coordinates mapped to their costs

.. py:data:: Index: int

  Index is represented in pyATF as an :code:`int` value.

.. py:class:: pyatf.search_techniques.search_technique_1d.SearchTechnique1D

  Searches over one-dimensional index space :math:`\{ 0 , ... , |SP|-1 \}`, where :math:`|SP|` is the search space size.

  .. py:function:: initialize(search_space_size: int)

    Initializes the search technique.

    :param search_space_size: the total number of configurations in the search space

  .. py:function:: finalize()

    Finalizes the search technique.

  .. py:function:: get_next_indices() -> Set[Index]

    Returns the next indices in :math:`\{ 0 , ... , |SP|-1 \}` for which the costs are requested.

    Function :code:`get_next_indices()` is called by pyATF before each call to :code:`report_costs(...)`.

    :return: indices in :math:`\{ 0 , ... , |SP|-1 \}`

  .. py:function:: report_costs(costs: Dict[Index, Cost])

    Processes costs for indices requested via function :code:`get_next_indices()`.

    Function :code:`report_costs(...)` is called by pyATF after each call to :code:`get_next_indices()`.

    :param costs: indices mapped to their costs
