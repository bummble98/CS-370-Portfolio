For this project we were tasked with designing an AI agent that would learn to traverse a maze in the most efficient way possible. Most of the code for this project was provided for us except for the center learning loops. This is the code I wrote myself

# For each epoch:
    for epoch in range(n_epoch):
        #    Agent_cell = randomly select a free cell
        Agent_cell = random.choice(qmaze.free_cells)
        #    Reset the maze with agent set to above position
        qmaze.reset(Agent_cell)
        #    Hint: Review the reset method in the TreasureMaze.py class.
        #    envstate = Environment.current_state
        envstate = qmaze.observe()
        #    Hint: Review the observe method in the TreasureMaze.py class.
        n_episodes = 0
        #    While state is not game over:
        game_status = 'not_over'
        while game_status == 'not_over':
            #        previous_envstate = envstate
            previous_envstate = envstate
            #        Action = randomly choose action (left, right, up, down) either by exploration or by exploitation
            if np.random.rand() < epsilon:
                action = np.random.choice(4)
            else:
                action = np.argmax(model.predict(previous_envstate)[0])
            #        envstate, reward, game_status = qmaze.act(action)
            #    Hint: Review the act method in the TreasureMaze.py class.
            envstate, reward, game_status = qmaze.act(action)
            #        episode = [previous_envstate, action, reward, envstate, game_status]
            episode = [previous_envstate, action, reward, envstate, game_status]
            #        Store episode in Experience replay object
            #    Hint: Review the remember method in the GameExperience.py class.
            experience.remember(episode)
            #        Train neural network model and evaluate loss
            #    Hint: Call GameExperience.get_data to retrieve training data (input and target) and pass to model.fit method 
            #          to train the model. You can call model.evaluate to determine loss.
            if n_episodes % 5 == 0 and n_episodes != 0:
                inputs, targets = experience.get_data()
                model.fit(inputs, targets, epochs=32, batch_size=64, verbose=0)
                loss = model.evaluate(inputs, targets, verbose=0)
            n_episodes += 1
            #    If the win rate is above the threshold and your model passes the completion check, that would be your epoch.
        if game_status == 'win':
            win_history.append(1)
        else:
            win_history.append(0)

        inputs, targets = experience.get_data()
        model.fit(inputs, targets, epochs=32, batch_size=64, verbose=0)
        loss = model.evaluate(inputs, targets, verbose=0)
        win_rate = sum(win_history)/len(win_history)
        
Connect your learning from throughout this course to the larger field of computer science:

What do computer scientists do and why does it matter?
What are my ethical responsibilities to the end user and the organization?
As computer scientists we work towards finding new ways to help aid the work people do everyday. Whether it be planning a schedule, sending a message, or just relaxing for the night with a game, computer science helps make it all happen. And because of this, we need to make sure we protect the users and ensure programs that are released are not only functional but safe. This means helping ensure the software is not harmful to their system, and protects the users privacy, while offering a service without any other malicious intent, especially as technology slowly creeps into a majority of everyones lives.

How do I approach a problem as a computer scientist?
For me, when approaching a problem, I like to take the perspective of the people it actually affects. If I want to design a program to help makeup artists design new routines, I would certainly need the opinion of makeup artists, since as a person who does not use makeup, I would have no clue on how to properly tackle the issue. This helps tackle the problem in a two pronged aproach, as it not only tells us how to approach the issue, but also how we could best design the application with the end users in mind.

