```bash
git clone git@github.com:HiroIshida/ARHMM.jl.git
] dev ./ARHMM.jl
```

```bash
cd ARHMM.jl
julia --project -e 'ENV["PTYHON"] = "/usr/bin/python"; import Pkg; import PyCall; Pkg.build("PyCall");'
julia --project -e 'import Pkg; Pkg.test()'
```

`` 
julia --project ./python/segment.jl 3
```
