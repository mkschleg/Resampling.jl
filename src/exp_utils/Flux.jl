module FluxUtil

using ..Reproduce
import Flux

function opt_settings!(as::ArgParseSettings)
    @add_arg_table as begin
        "--opt"
        arg_type=String
        required=true
        "--optparams"
        arg_type=Float64
        required=true
    end
end

function get_optimizer(parsed::Dict)
    kt = keytype(parsed)
    get_optimizer(parsed[kt("opt")], parsed[kt("optparams")])
end

function get_optimizer(opt_string::AbstractString, params)
    opt_func = getproperty(Flux, Symbol(opt_string))
    return opt_func(params...)
end


end
