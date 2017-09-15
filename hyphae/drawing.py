##############################
# DRAWING
####################
def draw_hyphae():
    """ Simple draw routine, each node as a circle """
    logging.debug("Drawing")
    #clear the context
    utils.clear_canvas(ctx)
    #from the root node of the graph
    nodes = deque([x['uuid'] for x in root_nodes])
    #BFS the tree
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        #todo: get a line between currentNode and predecessor
        #draw the node / line
        ctx.set_source_rgba(*colours[currentNode['colour']])
        logging.debug("Circle: {:.2f},  {:.2f}".format(*currentNode['loc']))
        utils.drawCircle(ctx, *currentNode['loc'], currentNode['d']-SIZE_DIFF)
        #get it's children
        nodes.extend(graph.successors(currentUUID))
            
    return True

def draw_hyphae_2():
    """ Draw an alternate form of the graph """
    logging.debug("Drawing alternate")
    utils.clear_canvas(ctx)
    nodes = deque([graph.successors(x['uuid']) for x in root_nodes])
    #BFS the tree:
    ctx.set_source_rgba(*MAIN_COLOUR)
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        prev = allNodes[graph.predecessors(currentNode['uuid'])[0]]
        points = utils.createLine(*currentNode['loc'], *prev['loc'], LINE_DISTORTION_UPSCALING)
        length_of_line = np.linalg.norm(points[-1] - points[0])
        distorted = utils.displace_along_line(points, \
                                              length_of_line * LINE_PROPORTION_DISTORTION, \
                                              LINE_DISTORTION_UPSCALING)
        nodes.extend(graph.successors(currentUUID))
        for x, y in distorted:
            utils.drawCircle(ctx, x, y, MIN_NODE_SIZE)
        #for x, y in points:
        #    utils.drawCircle(ctx, x, y, \
        #    utils.clamp(currentNode['d']-SIZE_DIFF, MIN_NODE_SIZE, NODE_START_SIZE))

    return True
    
def draw_hyphae_3():
    """ An alternate draw routine, drawing lines for branches """    
    utils.clear_canvas(ctx)
    nodes = deque([x['uuid'] for x in root_nodes])
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        branchUUID = get_branch_point(currentUUID)
        branchNode = allNodes[branchUUID]
        
        ctx.set_source_rgba(*colours[currentNode['colour']])
        ctx.set_line_width(LINE_WIDTH)
        utils.drawCircle(ctx, *currentNode['loc'], currentNode['d']-SIZE_DIFF)
        ctx.move_to(*currentNode['loc'])
        ctx.line_to(*branchNode['loc'])
        ctx.stroke()
        nodes.extend(graph.successors(branchNode['uuid']))
        for succUUID in graph.successors(branchNode['uuid']):
            succNode = allNodes[succUUID]
            ctx.move_to(*branchNode['loc'])
            ctx.line_to(*succNode['loc'])
            ctx.stroke()

def draw_hyphae_4():
    """ Another alternate draw routine, which paths wiggles """
    utils.clear_canvas(ctx)
    nodes = deque([x['uuid'] for x in root_nodes])
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        pathUUIDs = get_path(currentUUID)

        ctx.set_source_rgba(*colours[currentNode['colour']])
        ctx.set_line_width(LINE_WIDTH)
        utils.drawCircle(ctx, *currentNode['loc'], currentNode['d']-SIZE_DIFF)

        if len(pathUUIDs) == 0:
            nodes.extend(graph.successors(currentUUID))
            for succUUID in graph.successors(currentUUID):
                lastNode = allNodes[currentUUID]
                succNode = allNodes[succUUID]
                ctx.move_to(*lastNode['loc'])
                ctx.line_to(*succNode['loc'])
                ctx.stroke()
            continue
        
        ctx.move_to(*currentNode['loc'])
        for nextUUID in pathUUIDs:
            nextNode = allNodes[nextUUID]
            ctx.line_to(*nextNode['loc'])
        ctx.stroke()
        nodes.extend(graph.successors(pathUUIDs[-1]))            

        for succUUID in graph.successors(pathUUIDs[-1]):
            lastNode = allNodes[pathUUIDs[-1]]
            succNode = allNodes[succUUID]
            ctx.move_to(*lastNode['loc'])
            ctx.line_to(*succNode['loc'])
            ctx.stroke()
