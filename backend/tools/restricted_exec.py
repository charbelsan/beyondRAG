import ast, types, builtins, textwrap
SAFE_BUILTINS={k:v for k,v in builtins.__dict__.items() if k in ("None","True","False","float","int","str","list","dict","set","len","range")}
def call(code:str,tool_map:dict):
    """Exec a single-line 'tool(arg, …)' safely."""
    mod=ast.parse(textwrap.dedent(code),"<tool>",mode="exec")
    if len(mod.body)!=1 or not isinstance(mod.body[0],ast.Expr): raise ValueError("not a single expr")
    call_node=mod.body[0].value
    if not isinstance(call_node,ast.Call) or not isinstance(call_node.func,ast.Name):
        raise ValueError("must be Call")
    fname=call_node.func.id
    if fname not in tool_map: raise ValueError("forbidden tool")
    args=[ast.literal_eval(a) for a in call_node.args]
    kwargs={k.arg:ast.literal_eval(k.value) for k in call_node.keywords}
    return tool_map[fname](*args,**kwargs)
