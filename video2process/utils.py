def flatten_openapi(schema):
    try:
        defs = {}

        # Cover recursive submodels
        for key, value in schema['$defs'].items():
            replacement = value

            for pkey in value['properties']:
                if '$ref' in value['properties'][pkey]:
                    replacement['properties'][pkey] = defs[value['properties'][pkey]['$ref']]
                elif 'items' in value['properties'][pkey] and '$ref' in value['properties'][pkey]['items']:
                    replacement['properties'][pkey]['items'] = defs[value['properties'][pkey]['items']['$ref']]
            defs[f'#/$defs/{key}'] = replacement
    except KeyError:
        return schema

    for key in schema['properties']:
        # Replace direct ussage of submodel
        if '$ref' in schema['properties'][key]:
            ref = schema['properties'][key]['$ref']
            schema['properties'][key] = defs[ref]
        # Replace list of submodels
        elif 'items' in schema['properties'][key]:
            if '$ref' in schema['properties'][key]['items']:
                ref = schema['properties'][key]['items']['$ref']
                schema['properties'][key]['items'] = defs[ref]

    del schema['$defs']
    return schema