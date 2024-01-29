Search Techniques
=================

.. cpp:type:: std::vector<double> coordinates

  Coordinates are in ATF represented as a vector of :code:`double` s.

.. cpp:type:: double cost_t

  We currently use :code:`double` as cost type.

.. cpp:class:: search_technique

  Searches over multi-dimensional coordinate space :math:`(0,1]^D`.

  .. cpp:function:: initialize(size_t dimensionality)

    Initializes the search technique.

    :param dimensionality: "D" of the coordinate space

  .. cpp:function:: finalize()

    Finalizes the search technique.

  .. cpp:function:: std::set<coordinates> get_next_coordinates()

    Returns the next coordinates in :math:`(0,1]^D` for which the costs are requested.

    Function :code:`get_next_coordinates()` is called by ATF before each call to :code:`report_costs(...)`.

    :return: coordinates in :math:`(0,1]^D`

  .. cpp:function:: report_costs(const std::map<coordinates, cost_t> &costs)

    Processes costs for coordinates requested via function :code:`get_next_coordinates()`.

    Function :code:`report_costs(...)` is called by ATF after each call to :code:`get_next_coordinates()`.

    :param costs: coordinates mapped to their costs

.. cpp:type:: atf::big_int index

  Index is represented in ATF as an integer value (:code:`atf::big_int` is used exactly the same as :code:`int`).

.. cpp:class:: search_technique_1d

  Searches over one-dimensional index space :math:`\{ 0 , ... , |SP|-1 \}`, where :math:`|SP|` is the search space size.

  .. cpp:function:: initialize(atf::big_int search_space_size)

    Initializes the search technique.

    :param search_space_size: the total number of configurations in the search space

  .. cpp:function:: finalize()

    Finalizes the search technique.

  .. cpp:function:: std::set<index> get_next_indices()

    Returns the next indices in :math:`\{ 0 , ... , |SP|-1 \}` for which the costs are requested.

    Function :code:`get_next_indices()` is called by ATF before each call to :code:`report_costs(...)`.

    :return: indices in :math:`\{ 0 , ... , |SP|-1 \}`

  .. cpp:function:: report_costs(const std::map<index, cost_t> &costs)

    Processes costs for indices requested via function :code:`get_next_indices()`.

    Function :code:`report_costs(...)` is called by ATF after each call to :code:`get_next_indices()`.

    :param costs: indices mapped to their costs
