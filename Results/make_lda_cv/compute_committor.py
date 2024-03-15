def b_before_a(label_trajectory, state_a=0, state_b=1):
    # save the committor labels
    import numpy as np
    locations_a = np.where(label_trajectory==state_a)[0]
    locations_b = np.where(label_trajectory==state_b)[0]
    if len(locations_a) == 0:
        locations_a= np.array([-1])
    if len(locations_b) == 0:
        locations_b = np.array([-1])
    
    final_known_state = max(locations_a.max(), locations_b.max())

    locations_a_iter = iter(locations_a)
    locations_b_iter = iter(locations_b)

    first_a = next(locations_a_iter)
    first_b = next(locations_b_iter)

    commitor = np.zeros( len(label_trajectory) )
    current_index = 0
    while first_a > -1 and first_b > -1:
        if first_b < first_a: 
            commitor[current_index:first_b+1] = 1
            current_index = first_b
            first_b = next(locations_b_iter, -1)
        else:
            current_index = first_a + 1
            first_a = next(locations_a_iter, -2)

    if first_a < 0:
        commitor[current_index:max(locations_b)+1] = 1

    commitor[final_known_state+1:] = -1

    return commitor

