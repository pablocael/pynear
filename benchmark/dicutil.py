def set_dict(dic, key_path, value):

    d = dic
    for p in key_path[:-1]:
        d[p] = d.get(p, {})
        d = d[p]
        d[key_path[-1]] = value
    return dic

def get_dict(dic, key_path):
    return reduce(lambda d, k: d[k], key_path, dic)
