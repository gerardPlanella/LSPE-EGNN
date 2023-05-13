
def get_pe_attribute(pe_name):
    if pe_name == 'rw':
        return 'random_walk_pe'
    elif pe_name == 'lap':
        return 'laplacian_eigenvector_pe'
    else:
        raise NotImplementedError(f"PE method \"{pe_name}\" not implemented.")
