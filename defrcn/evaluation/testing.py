import logging
import numpy as np
import pprint
import sys
from collections import Mapping, OrderedDict


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    """
    assert isinstance(results, OrderedDict), results  # unordered results cannot be properly printed
    logger = logging.getLogger(__name__)
    for task, res in results.items():
        # Don't print "AP-category" metrics since they are usually not tracked.
        if task == "attr":
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            logger.info("copypaste: Task: {}".format(task))

            det_class_metrics = []
            det_class_base_metrics = []
            det_class_novel_metrics = []
            attr_class_metrics = []
            attr_class_base_metrics = []
            attr_class_novel_metrics = []
            cluster_top_metrics = {}
            cluster_base_top_metrics = {}
            cluster_novel_top_metrics = {}
            cluster_ac_metrics = []

            for k, v in important_res:
                if k.startswith("det_class_"):
                    if k.startswith("det_class_base_"):
                        det_class_base_metrics.append((k, v))
                    elif k.startswith("det_class_novel_"):
                        det_class_novel_metrics.append((k, v))
                    else:
                        det_class_metrics.append((k, v))
                elif k.startswith("attr_class_"):
                    if k.startswith("attr_class_base_"):
                        attr_class_base_metrics.append((k, v))
                    elif k.startswith("attr_class_novel_"):
                        attr_class_novel_metrics.append((k, v))
                    else:
                        attr_class_metrics.append((k, v))
                elif k.startswith("attr_cluster_top"):
                    cluster_top_metrics[k] = v
                elif k.startswith("attr_cluster_base_top"):
                    cluster_base_top_metrics[k] = v
                elif k.startswith("attr_cluster_novel_top"):
                    cluster_novel_top_metrics[k] = v
                elif k.startswith("ac") and "_t" in k:
                    cluster_ac_metrics.append((k, v))

            if det_class_metrics:
                det_top_keys = [
                    "det_class_top1_acc",
                    "det_class_top2_acc",
                    "det_class_top3_acc",
                ]
                det_class_metrics = [
                    (k, v) for k, v in det_class_metrics if k in det_top_keys
                ]
                if det_class_metrics:
                    det_class_metrics.sort(key=lambda item: det_top_keys.index(item[0]))
                    logger.info("copypaste: {} - det_class".format(task) + "\n"+ \
                        ",".join([k[0] for k in det_class_metrics]) + "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in det_class_metrics]))

                det_rec_keys = [
                    "det_class_top1_rec",
                    "det_class_top2_rec",
                    "det_class_top3_rec",
                ]
                det_class_rec_metrics = [
                    (k, v) for k, v in important_res if k in det_rec_keys
                ]
                det_class_rec_metrics.sort(key=lambda item: det_rec_keys.index(item[0]))
                if det_class_rec_metrics:
                    logger.info("copypaste: {} - det_class_rec".format(task) + "\n"+ \
                        ",".join([k[0] for k in det_class_rec_metrics]) + "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in det_class_rec_metrics]))

            if det_class_base_metrics:
                det_base_keys = [
                    "det_class_base_top1_acc",
                    "det_class_base_top2_acc",
                    "det_class_base_top3_acc",
                ]
                det_class_base_metrics = [
                    (k, v) for k, v in det_class_base_metrics if k in det_base_keys
                ]
                if det_class_base_metrics:
                    det_class_base_metrics.sort(key=lambda item: det_base_keys.index(item[0]))
                    logger.info("copypaste: {} - det_class_base".format(task)+ "\n"+ \
                        ",".join([k[0] for k in det_class_base_metrics])+ "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in det_class_base_metrics]))

                det_base_rec_keys = [
                    "det_class_base_top1_rec",
                    "det_class_base_top2_rec",
                    "det_class_base_top3_rec",
                ]
                det_class_base_rec_metrics = [
                    (k, v) for k, v in important_res if k in det_base_rec_keys
                ]
                det_class_base_rec_metrics.sort(key=lambda item: det_base_rec_keys.index(item[0]))
                if det_class_base_rec_metrics:
                    logger.info("copypaste: {} - det_class_base_rec".format(task)+ "\n"+ \
                        ",".join([k[0] for k in det_class_base_rec_metrics])+ "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in det_class_base_rec_metrics]))

            if det_class_novel_metrics:
                det_novel_keys = [
                    "det_class_novel_top1_acc",
                    "det_class_novel_top2_acc",
                    "det_class_novel_top3_acc",
                ]
                det_class_novel_metrics = [
                    (k, v) for k, v in det_class_novel_metrics if k in det_novel_keys
                ]
                if det_class_novel_metrics:
                    det_class_novel_metrics.sort(key=lambda item: det_novel_keys.index(item[0]))
                    logger.info("copypaste: {} - det_class_novel".format(task)+ "\n"+ \
                        ",".join([k[0] for k in det_class_novel_metrics])+ "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in det_class_novel_metrics]))

                det_novel_rec_keys = [
                    "det_class_novel_top1_rec",
                    "det_class_novel_top2_rec",
                    "det_class_novel_top3_rec",
                ]
                det_class_novel_rec_metrics = [
                    (k, v) for k, v in important_res if k in det_novel_rec_keys
                ]
                det_class_novel_rec_metrics.sort(key=lambda item: det_novel_rec_keys.index(item[0]))
                if det_class_novel_rec_metrics:
                    logger.info("copypaste: {} - det_class_novel_rec".format(task)+ "\n"+ \
                        ",".join([k[0] for k in det_class_novel_rec_metrics])+ "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in det_class_novel_rec_metrics]))

            if attr_class_metrics:
                class_top_keys = [
                    "attr_class_top1_acc",
                    "attr_class_top2_acc",
                    "attr_class_top3_acc",
                ]
                attr_class_metrics = [
                    (k, v) for k, v in attr_class_metrics if k in class_top_keys
                ]
                if attr_class_metrics:
                    attr_class_metrics.sort(key=lambda item: class_top_keys.index(item[0]))
                    logger.info("copypaste: {} - attr_class".format(task) + "\n"+ \
                        ",".join([k[0] for k in attr_class_metrics]) + "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in attr_class_metrics]))

                attr_rec_keys = [
                    "attr_class_top1_rec",
                    "attr_class_top2_rec",
                    "attr_class_top3_rec",
                ]
                attr_class_rec_metrics = [
                    (k, v) for k, v in important_res if k in attr_rec_keys
                ]
                attr_class_rec_metrics.sort(key=lambda item: attr_rec_keys.index(item[0]))
                if attr_class_rec_metrics:
                    logger.info("copypaste: {} - attr_class_rec".format(task) + "\n"+ \
                        ",".join([k[0] for k in attr_class_rec_metrics]) + "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in attr_class_rec_metrics]))

            if attr_class_base_metrics:
                base_top_keys = [
                    "attr_class_base_top1_acc",
                    "attr_class_base_top2_acc",
                    "attr_class_base_top3_acc",
                ]
                attr_class_base_metrics = [
                    (k, v) for k, v in attr_class_base_metrics if k in base_top_keys
                ]
                if attr_class_base_metrics:
                    attr_class_base_metrics.sort(key=lambda item: base_top_keys.index(item[0]))
                    logger.info("copypaste: {} - attr_class_base".format(task)+ "\n"+ \
                        ",".join([k[0] for k in attr_class_base_metrics])+ "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in attr_class_base_metrics]))

                attr_base_rec_keys = [
                    "attr_class_base_top1_rec",
                    "attr_class_base_top2_rec",
                    "attr_class_base_top3_rec",
                ]
                attr_class_base_rec_metrics = [
                    (k, v) for k, v in important_res if k in attr_base_rec_keys
                ]
                attr_class_base_rec_metrics.sort(key=lambda item: attr_base_rec_keys.index(item[0]))
                if attr_class_base_rec_metrics:
                    logger.info("copypaste: {} - attr_class_base_rec".format(task)+ "\n"+ \
                        ",".join([k[0] for k in attr_class_base_rec_metrics])+ "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in attr_class_base_rec_metrics]))

            if attr_class_novel_metrics:
                novel_top_keys = [
                    "attr_class_novel_top1_acc",
                    "attr_class_novel_top2_acc",
                    "attr_class_novel_top3_acc",
                ]
                attr_class_novel_metrics = [
                    (k, v) for k, v in attr_class_novel_metrics if k in novel_top_keys
                ]
                if attr_class_novel_metrics:
                    attr_class_novel_metrics.sort(key=lambda item: novel_top_keys.index(item[0]))
                    logger.info("copypaste: {} - attr_class_novel".format(task)+ "\n"+ \
                        ",".join([k[0] for k in attr_class_novel_metrics])+ "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in attr_class_novel_metrics]))

                attr_novel_rec_keys = [
                    "attr_class_novel_top1_rec",
                    "attr_class_novel_top2_rec",
                    "attr_class_novel_top3_rec",
                ]
                attr_class_novel_rec_metrics = [
                    (k, v) for k, v in important_res if k in attr_novel_rec_keys
                ]
                attr_class_novel_rec_metrics.sort(key=lambda item: attr_novel_rec_keys.index(item[0]))
                if attr_class_novel_rec_metrics:
                    logger.info("copypaste: {} - attr_class_novel_rec".format(task)+ "\n"+ \
                        ",".join([k[0] for k in attr_class_novel_rec_metrics])+ "\n"+ \
                        ",".join(["{0:.4f}".format(k[1]) for k in attr_class_novel_rec_metrics]))

            top_keys = [
                "attr_cluster_top1_acc",
                "attr_cluster_top2_acc",
                "attr_cluster_top3_acc",
            ]
            top_metrics = [(k, cluster_top_metrics[k]) for k in top_keys if k in cluster_top_metrics]
            if cluster_top_metrics or cluster_ac_metrics or cluster_base_top_metrics or cluster_novel_top_metrics:
                logger.info("copypaste: {} - attr_cluster".format(task)+ "\n"+ \
                    ",".join([k[0] for k in top_metrics])+ "\n"+ \
                    ",".join(["{0:.4f}".format(k[1]) for k in top_metrics]))

            base_top_keys = [
                "attr_cluster_base_top1_acc",
                "attr_cluster_base_top2_acc",
                "attr_cluster_base_top3_acc",
            ]
            base_top_metrics = [
                (k, cluster_base_top_metrics[k])
                for k in base_top_keys
                if k in cluster_base_top_metrics
            ]
            if base_top_metrics:
                logger.info("copypaste: attr_cluster - base"+ "\n"+ \
                    ",".join([k[0] for k in base_top_metrics])+ "\n"+ \
                    ",".join(["{0:.4f}".format(k[1]) for k in base_top_metrics]))

            novel_top_keys = [
                "attr_cluster_novel_top1_acc",
                "attr_cluster_novel_top2_acc",
                "attr_cluster_novel_top3_acc",
            ]
            novel_top_metrics = [
                (k, cluster_novel_top_metrics[k])
                for k in novel_top_keys
                if k in cluster_novel_top_metrics
            ]
            if novel_top_metrics:
                logger.info("copypaste: attr_cluster - novel"+ "\n"+ \
                    ",".join([k[0] for k in novel_top_metrics])+ "\n"+ \
                    ",".join(["{0:.4f}".format(k[1]) for k in novel_top_metrics]))

            def _parse_ac_key(name):
                # ac{n}_t{m} -> (n, m)
                try:
                    prefix, suffix = name.split("_t")
                    index = int(prefix[2:])
                    t = int(suffix)
                    return index, t
                except ValueError:
                    return None

            if cluster_ac_metrics:
                parsed = []
                for k, v in cluster_ac_metrics:
                    key = _parse_ac_key(k)
                    if key is not None:
                        parsed.append((key, k, v))
                parsed.sort(key=lambda item: (item[0][1], item[0][0]))
                attr_cluster_acc = ""
                for t in (1, 2, 3):
                    metrics = [(k, v) for (idx, k, v) in parsed if idx[1] == t]
                    if not metrics:
                        continue
                    attr_cluster_acc += " attr_cluster - ac_t{}".format(t) + '\n'
                    for idx in range(0, len(metrics), 10):
                        chunk = metrics[idx : idx + 10]
                    attr_cluster_acc += ",".join([k[0] for k in chunk]) + '\n'
                    attr_cluster_acc += ",".join(["{0:.4f}".format(k[1]) for k in chunk]) + '\n'
                logger.info(attr_cluster_acc)
        else:
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            logger.info("copypaste: Task: {}".format(task))
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
    for task, res in results.items(): # tag add
        if task == "bbox":
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            logger.info("copypaste: Task: {}".format(task))
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))


def verify_results(cfg, results):
    """
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}

    Returns:
        bool: whether the verification succeeds or not
    """
    expected_results = cfg.TEST.EXPECTED_RESULTS
    if not len(expected_results):
        return True

    ok = True
    for task, metric, expected, tolerance in expected_results:
        actual = results[task][metric]
        if not np.isfinite(actual):
            ok = False
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False

    logger = logging.getLogger(__name__)
    if not ok:
        logger.error("Result verification failed!")
        logger.error("Expected Results: " + str(expected_results))
        logger.error("Actual Results: " + pprint.pformat(results))

        sys.exit(1)
    else:
        logger.info("Results verification passed.")
    return ok


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r
