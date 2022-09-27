import turtle 

wind = turtle.Screen()
wind.title("Ping Pong By Manar")
wind.bgcolor("white")
wind.setup(width=800, height=600)
wind.tracer(0)

score1=200
score2=200
#shapes
scoreBord =turtle.Turtle()
scoreBord.shape("square")
scoreBord.color("red")
scoreBord.shapesize(stretch_wid=6,stretch_len=10)
scoreBord.penup()
scoreBord.goto(0,300)
scoreBord.write('Manar Gamal',font=('Arial',8,'normal'))

madrab1 =turtle.Turtle()
madrab1.speed(0)
madrab1.shape("square")
madrab1.color("pink")
madrab1.shapesize(stretch_wid=5,stretch_len=1)
madrab1.penup()
madrab1.goto(-350,0)

madrab2 =turtle.Turtle()
madrab2.speed(0)
madrab2.shape("square")
madrab2.color("gray")
madrab2.shapesize(stretch_wid=5,stretch_len=1)
madrab2.penup()
madrab2.goto(350,0)

ball =turtle.Turtle()
ball.speed(0)
ball.shape("square")
ball.color("black")
ball.penup()
ball.goto(0,0)
ball.dx =2
ball.dy =2

#functions
def madrab1Up():
   y=madrab1.ycor()
   y +=20
   madrab1.sety(y)

def madrab1Down():
    y=madrab1.ycor()
    y-=20
    madrab1.sety(y)

def madrab2Up():
   y=madrab2.ycor()
   y +=20
   madrab2.sety(y)

def madrab2Down():
    y=madrab2.ycor()
    y-=20
    madrab2.sety(y)

#keyboard bindings
wind.listen()
wind.onkeypress(madrab1Up,"w")
wind.onkeypress(madrab1Down,"s")

wind.onkeypress(madrab2Up,"Up")
wind.onkeypress(madrab2Down,"Down")


#main game loop
while True:
    wind.update()

    ball.setx(ball.xcor()+ball.dx)
    ball.sety(ball.ycor()+ball.dy)

    if ball.ycor() >290:
        ball.sety(290)
        ball.dy *=-1

    if ball.ycor() <-290:
        ball.sety(-290)
        ball.dy *=-1
    
    if ball.xcor()>390:
        ball.goto(0,0)
        ball.dx *=-1
    
    if ball.xcor()<-390:
        ball.goto(0,0)
        ball.dx *= -1

    if((ball.xcor() > 340 and ball.xcor() < 350) and (ball.ycor() < madrab2.ycor() + 40 and ball.ycor() > madrab2.ycor() - 40)):
        ball.setx(340)
        ball.dx *= -1
    
      
    if((ball.xcor() < -340 and ball.xcor()>-350) and (ball.ycor() < madrab1.ycor() + 40 and ball.ycor() > madrab1.ycor() - 40)):
        ball.setx(-340)
        ball.dx *=-1
