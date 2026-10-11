"""Dimension-only extension of the repository's exact batch token contract."""
import ast
import inspect
import textwrap

import numpy as np


def extend_converter(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    function.name = 'batch_all_n_tokens'
    function.args.args.append(ast.arg(arg='N'))
    counts = {'shape': 0, 'slice': 0, 'reshape': 0, 'top': 0}

    def expression(source):
        return ast.parse(source, mode='eval').body

    def matches(node, source):
        return ast.dump(node) == ast.dump(expression(source))

    class DimensionsOnly(ast.NodeTransformer):
        def visit_Tuple(self, node):
            if ast.dump(node) == ast.dump(ast.parse('(B, 8, 13)', mode='eval').body):
                node.elts[1] = ast.parse('3 + N', mode='eval').body
                counts['shape'] += 1
            return self.generic_visit(node)

        def visit_Subscript(self, node):
            if matches(node, 'state_batch[:, 9:34]'):
                counts['slice'] += 1
                return ast.copy_location(expression('state_batch[:, 9:]'), node)
            if matches(node, 'sorted_indices[:5]'):
                counts['top'] += 1
                return ast.copy_location(expression('sorted_indices[:N]'), node)
            return self.generic_visit(node)

        def visit_Call(self, node):
            # Children must be visited before matching the now-expanded slice.
            node = self.generic_visit(node)
            if matches(node, 'state_batch[:, 9:].reshape(B, 5, 5)'):
                node.args[1] = ast.Name(id='N', ctx=ast.Load())
                counts['reshape'] += 1
            return node

    tree = ast.fix_missing_locations(DimensionsOnly().visit(tree))
    assert counts == {'shape': 1, 'slice': 1, 'reshape': 1, 'top': 1}, counts
    namespace = {'np': np}
    exec(compile(tree, '<dimension-only-all-N-contract>', 'exec'), namespace)
    return namespace[function.name], ast.dump(tree)


def state_vector(robot, humans):
    r = np.asarray(robot, dtype=np.float32)
    h = np.asarray(humans, dtype=np.float64).reshape(-1)
    assert r.shape == (9,) and h.size % 5 == 0 and h.size > 0
    return np.concatenate([r, h])


def candidate_windows(history, next_tokens, length=24):
    # Exactly the scorer's prefix truncation and repeat-first left padding.
    hist = np.asarray(history, dtype=np.float32)
    assert hist.ndim == 3 and len(hist) > 0
    if len(hist) >= length - 1:
        prefix = hist[-(length - 1):]
    else:
        prefix = np.concatenate([np.repeat(hist[:1], length - 1 - len(hist), axis=0), hist])
    return np.ascontiguousarray(np.concatenate([
        np.broadcast_to(prefix[None], (len(next_tokens),) + prefix.shape), next_tokens[:, None]], axis=1))
