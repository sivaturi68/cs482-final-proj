rows = []

with open('./TestOut/exp2/detection_results.csv') as f:
    lines = f.readlines()
    for line in lines[1:]:
        fid, cid, conf, x, y, w, h = line.split(',')
        fid = int(fid.split('_')[-1])
        cid = int(cid)
        conf, x, y, w, h = [float(t) for t in (conf, x, y, w, h)]
